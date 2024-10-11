from lxml import html
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time


def initialize_browser():
    chrome_driver_path = "C:\\Users\\valer\\Downloads\\chromedriver-win64\\chromedriver.exe"
    chrome_options = Options()

    chrome_options.add_argument("--headless")

    chrome_service = Service(chrome_driver_path)

    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
    return driver

def close_browser(driver):
    driver.quit()

def download_html_dynamic(driver, url):
    driver.get(url)
    time.sleep(1)
    page_source = driver.page_source
    return page_source

def get_tank_html(driver, url):
    base_url = 'https://tanks.gg'
    full_url = base_url + url

    tank_html = download_html_dynamic(driver, full_url)
    if tank_html:
        print(f"Страница {full_url} загружена успешно.")
        return tank_html
    else:
        print(f"Ошибка при загрузке страницы {full_url}.")
        return None

def get_tank_list_url_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    tree = html.fromstring(html_content)

    tanks_block = tree.xpath('//div[@class="tanks"]')[0]

    tank_links = tanks_block.xpath('.//a')

    tanks_data = []
    for tank in tank_links:
        tank_url = tank.get('href')
        tanks_data.append(tank_url)

    return tanks_data

def get_tank_data(tank_html):
    tree = html.fromstring(tank_html)
    tank_name = tree.xpath('//div[@class="tank"]/h1/text()')[0].strip()
    #tank_image = tree.xpath('//div[@class="tank"]/img/@src')[0].strip()
    #if tank_image:

    parts = tree.xpath('//div[@class="tank"]/h1/small/text()')[0].strip().split()
    if len(parts) > 5:
        del parts[2]
        if len(parts) > 6:
            del parts[2]

    dpm = tree.xpath('//div[@class="mb-3 mb-md-0 ps-xxl-0 col-xxl-auto"]//div[@class="card-body"]//div[label[contains(text(), "DPM")]]/span/text()')[0].strip().replace(',', '')
    damage = tree.xpath('//div[@class="mb-3 mb-md-0 ps-xxl-0 col-xxl-auto"]//div[@class="card-body"]//div[label[contains(text(), "Damage")]]/span/text()')[0].strip()
    dispersion = tree.xpath('//div[@class="mb-3 mb-md-0 ps-xxl-0 col-xxl-auto"]//div[@class="card-body"]//div[label[contains(text(), "Dispersion")]]/span/text()')[0].strip()
    caliber = tree.xpath('//div[@class="mb-3 mb-md-0 ps-xxl-0 col-xxl-auto"]//div[@class="card-body"]//div[label[contains(text(), "Caliber")]]/span/text()')[0].strip()
    top_speed = tree.xpath('//div[@class="col-xxl-auto"]//div[@class="card-body"]//div[@class="stat-line highlight"]/span/text()')[0].strip()
    hp = tree.xpath('//div[@class="position-relative col-xxl-auto"]//div[@class="card-body"]//div[label[text()="Health"]]/span/text()')[0].strip().replace(',', '')
    total_weight = tree.xpath('//div[@class="position-relative col-xxl-auto"]//div[@class="card-body"]//div[label[contains(text(), "Total weight")]]/span/text()')[0].strip().replace(',', '')
    stacionary_camo = tree.xpath('//div[@class="position-relative col-xxl-auto"]//div[@class="card-body"]//div[label[contains(text(), "Stationary camo")]]/span/text()')[0].split('/')[0].strip()
    moving_camo = tree.xpath('//div[@class="position-relative col-xxl-auto"]//div[@class="card-body"]//div[label[contains(text(), "Moving camo")]]/span/text()')[0].split('/')[0].strip()
    view_range = tree.xpath('//div[@class="position-relative col-xxl-auto"]//div[@class="card-body"]//div[label[contains(text(), "View range")]]/span/text()')[0].strip()
    tank_cost = tree.xpath('//div[@class="position-relative col-xxl-auto"]//div[@class="card-body"]//div[label[contains(text(), "Tank cost")]]/span/var/text()')[0].strip().replace(',', '')

    print("Name:", tank_name, "\ntier:", parts[1], "\nnation:", parts[2], "\ntype:", parts[3] + ' ' + parts[4],
          "\nDamage:", damage, "\nDPM:", dpm, "\nDispersion:", dispersion, "\nCaliber (mm):", caliber,
          "\nTop speed (km/h):", top_speed, "\nHP:", hp, "\nTotal weight:", total_weight,
          "\nStationary camo:", stacionary_camo, "\nMoving camo:", moving_camo,
          "\nView range:", view_range, "\nTank cost:", tank_cost)

    return {
        'Name': tank_name,
        'Lvl': parts[1],
        'Nation': parts[2],
        'Tank_type': parts[3] + ' ' + parts[4],
        'DPM': dpm,
        'Damage': damage,
        'Dispersion': dispersion,
        'Caliber (mm)': caliber,
        'Top_speed (km/h)': top_speed,
        'HP': hp,
        'Total weight': total_weight,
        'Stacionary_camo (%)': stacionary_camo,
        'Moving_camo (%)': moving_camo,
        'View_range (m)': view_range,
        'Tank_cost': tank_cost            #Нужно ещё будет как-то поработать с золотом/серебром
    }



if __name__ == '__main__':
    tanks_data = get_tank_list_url_from_html("tank_list_dynamic.html")
    all_tanks_data = []
    web_driver = initialize_browser()

    for tank_url in tanks_data:
        print(f"Обрабатываю танк {tank_url}...")
        tank_html = get_tank_html(web_driver, tank_url)

        if tank_html:
            tank_info = get_tank_data(tank_html)
            all_tanks_data.append(tank_info)
            print(f"Данные о танке {tank_info['Name']} успешно добавлены.")

    close_browser(web_driver)