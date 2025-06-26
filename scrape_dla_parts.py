import os
import json
import base64
import re
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, InvalidSessionIdException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

OUTPUT_DIR = Path("scraped_parts")
OUTPUT_DIR.mkdir(exist_ok=True)
CSV_METADATA_FILE = OUTPUT_DIR / "metadata.csv"

if not CSV_METADATA_FILE.exists():
    with open(CSV_METADATA_FILE, "w") as f:
        f.write("page,nsn,part_number,cage,name,product_url,cover,support_images\n")

def setup_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("user-agent=Mozilla/5.0")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def sanitize(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name.strip())

def save_image(driver, img_element, save_path):
    try:
        src = img_element.get_attribute("src")
        if not src or "view.gif" in src:
            return False
        img_data = driver.execute_script("""
            const img = arguments[0];
            const canvas = document.createElement('canvas');
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            return canvas.toDataURL('image/png').substring(22);
        """, img_element)
        with open(save_path, "wb") as f:
            f.write(base64.b64decode(img_data))
        return True
    except Exception as e:
        print(f"⚠️ Error saving image: {e}")
        return False

def scrape_part(driver, part_url, nsn_from_main_page, page_num, retries=3):
    for attempt in range(retries):
        try:
            driver.get(part_url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".DNNModuleContent table")))
            break
        except Exception as e:
            print(f"🔁 Retry {attempt + 1} for: {part_url}")
            time.sleep(2)
            if attempt == retries - 1:
                print(f"❌ Failed after retries: {part_url}")
                return None

    metadata = {}
    try:
        table = driver.find_element(By.CSS_SELECTOR, ".DNNModuleContent table")
        rows = table.find_elements(By.TAG_NAME, "tr")
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) >= 2:
                key = cells[0].text.strip().rstrip(":")
                val = cells[1].text.strip()
                metadata[key] = val
    except Exception as e:
        print(f"⚠️ Metadata extraction failed: {e}")

    nsn = metadata.get("National Stock Number (NSN)", nsn_from_main_page).replace(" ", "").replace("/", "-").strip()
    part_number = metadata.get("Part Number", "no_part").strip()
    cage = metadata.get("CAGE Code", "nocage").strip()
    name = metadata.get("Name", "Unnamed Part").strip()

    folder_name = sanitize(f"{name}_{nsn}_{page_num}")[:100]
    folder = OUTPUT_DIR / folder_name

    if folder.exists():
        print(f"⚠️ Skipping duplicate folder: {folder_name}")
        return None
    folder.mkdir(exist_ok=True)

    image_files = []
    rows = driver.find_elements(By.CSS_SELECTOR, ".DNNModuleContent table tr")
    img_counter = 0
    for row in rows:
        try:
            img_tags = row.find_elements(By.TAG_NAME, "img")
            for img_tag in img_tags:
                image_name = "cover.png" if img_counter == 0 else f"support_{img_counter}.png"
                save_path = folder / image_name
                if save_image(driver, img_tag, save_path):
                    image_files.append(image_name)
                    img_counter += 1
        except Exception:
            continue

    structured = {
        "NSN": nsn,
        "Part Number": part_number,
        "CAGE": cage,
        "Name": name,
        "Product URL": part_url,
        "Images": {
            "cover": image_files[0] if image_files else None,
            "support": image_files[1:] if len(image_files) > 1 else []
        }
    }

    with open(folder / "metadata.json", "w") as f:
        json.dump(structured, f, indent=2)

    with open(CSV_METADATA_FILE, "a") as f:
        f.write(f"{page_num},{nsn},{part_number},{cage},{name},{part_url},{image_files[0] if image_files else ''},{'|'.join(image_files[1:])}\n")

    print(f"✅ Saved: {folder_name} | Images: {len(image_files)}")
    return folder_name

def run_all():
    for i in range(1, 17):  # Pages 1 to 16
        try:
            driver = setup_driver()
            url = f"https://www.dla.mil/Aviation/Offers/Engineering/RPPOB/Parts-Catalog/?udt_98399_param_page={i}"
            print(f"\n🌐 Scraping Page {i}: {url}")
            driver.get(url)

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".dnnGridItem td a"))
            )
            rows = driver.find_elements(By.CSS_SELECTOR, ".dnnGridItem")
            part_links = []
            seen_links = set()

            for row in rows:
                try:
                    link_el = row.find_element(By.CSS_SELECTOR, "td a[href*='_param_detail=']")
                    nsn_el = row.find_element(By.CSS_SELECTOR, "td:nth-child(1) a")
                    part_url = link_el.get_attribute("href")
                    nsn_text = nsn_el.text.strip().replace(" ", "").replace("/", "-")
                    if part_url and part_url not in seen_links:
                        part_links.append((part_url, nsn_text))
                        seen_links.add(part_url)
                except Exception as e:
                    print(f"⚠️ Error reading row: {e}")

            for link, nsn in part_links:
                try:
                    scrape_part(driver, link, nsn, i)
                    time.sleep(1)
                except Exception as e:
                    print(f"❌ Failed to scrape part URL: {link} | Error: {e}")

        except (InvalidSessionIdException, WebDriverException) as e:
            print(f"❌ Driver error. Skipping page {i}. Error: {e}")
            continue
        finally:
            driver.quit()
            time.sleep(3)
            print("🚀 Page scraping complete.")

if __name__ == "__main__":
    run_all()
