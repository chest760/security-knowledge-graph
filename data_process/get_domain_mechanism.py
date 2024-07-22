import requests
from typing import Any
from bs4 import BeautifulSoup, Tag
import pandas as pd
import time

class CAPEC_Scraping:
    def __init__(self) -> None:
        pass
    
    def find_domain(self, domain: Tag) -> str:
        row = domain.select("td")[1]
        text = row.get_text()
        return text
    
    def find_mechanism(self, mechanism: Tag) -> str:
        row = mechanism.select("td")[1]
        text = row.get_text()
        return text

    
    def find_relation_table(self):
        domain_and_mechanism_table = self.soup.select("div#Relationships")[0].select("div.indent")[0].select("div.indent")[-1]
        tr = domain_and_mechanism_table.select("tr")
        domain = tr[1]
        mechanism = tr[2]
        domain = self.find_domain(domain)
        mechanism = self.find_domain(mechanism)
        
        return domain, mechanism
        
    def __call__(self, url: str) -> Any:
        # スクレイピング対象の URL にリクエストを送り HTML を取得する
        self.res = requests.get(url)

        # レスポンスの HTML から BeautifulSoup オブジェクトを作る
        self.soup = BeautifulSoup(self.res.text, 'html.parser')
        
        domain, mechanism = self.find_relation_table()
        
        return domain, mechanism
        
        


if __name__ == "__main__":
    capec_ids = pd.read_csv("../data/raw/capec.csv")["ID"]
    scraper = CAPEC_Scraping()
    row = []
    domain, mechanism = scraper(f"https://capec.mitre.org/data/definitions/690.html")
    for id in capec_ids:
        try:
            domain, mechanism = scraper(f"https://capec.mitre.org/data/definitions/{id}.html")
            row.append([id, domain, mechanism])
            print(id)
            time.sleep(3)
        except Exception as e:
            print(f"{id}: Error")
            print(e)
            continue
    
    df = pd.DataFrame(data=row, columns=["ID", "Domain", "Mechanism"])
    df.to_csv("../data/raw/domain_mechanism.csv")
    