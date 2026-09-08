import requests
import os

# SEC requires a User-Agent header identifying who you are
HEADERS = {"User-Agent": "Lucas Loketek lucasloketek@gmail.com"}

# The three banks and their SEC CIK numbers
BANKS = {
    "jpmorgan": "0000019617",
    "bankofamerica": "0000070858",
    "wellsfargo": "0000072971",
}

os.makedirs("documents", exist_ok=True)

def get_latest_10k(cik):
    # Get the company's filing history from SEC's official data API
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = requests.get(url, headers=HEADERS).json()

    recent = data["filings"]["recent"]
    # Find the most recent filing where the form type is "10-K"
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            accession = recent["accessionNumber"][i].replace("-", "")
            doc = recent["primaryDocument"][i]
            cik_int = int(cik)
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{doc}"
            return doc_url
    return None

for name, cik in BANKS.items():
    print(f"Fetching {name}...")
    doc_url = get_latest_10k(cik)
    if doc_url:
        html = requests.get(doc_url, headers=HEADERS).text
        out_path = f"documents/{name}_10k.html"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  Saved {out_path} ({len(html)//1000} KB)")
    else:
        print(f"  No 10-K found for {name}")

print("Done.")