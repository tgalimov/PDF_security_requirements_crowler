from googleapiclient.discovery import build
import config

service = build("customsearch", "v1",
                developerKey=config.developerKey)


def google_search(request_row, n):
    res = service.cse().list(
        q=request_row,
        cx='017576662512468239146:omuauf_lfve',
        num=10,
        start=n * 10,
    ).execute()
    return res


def google_search_next_pages(request_row):
    # res = google_search(request_row)
    service.cse().list(
        q=request_row,
        cx='017576662512468239146:omuauf_lfve',
        num=10,
    ).execute()


def search_pdf(request_row, number_of_pages):
    data = []
    for page in range(number_of_pages):
        res = google_search(request_row, page)
        if "items" in res:
            links_list = res["items"]
            data.extend(links_list)
        else:
            continue
    links = list(map(lambda el: el["link"], data))
    filtered_list = list(filter(lambda cn: cn.endswith(".pdf"), links))
    return filtered_list

