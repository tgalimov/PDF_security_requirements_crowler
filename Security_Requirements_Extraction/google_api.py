from googleapiclient.discovery import build


def search_pdf(request_row):
    # Build a service object for interacting with the API. Visit
    # the Google APIs Console <http://code.google.com/apis/console>
    # to get an API key for your own application.
    service = build("customsearch", "v1",
                    developerKey="AIzaSyDDYOkM9cjUJcD_iXBPjmTQJERRD5hJ7H0")

    res = service.cse().list(
        q=request_row,
        cx='017576662512468239146:omuauf_lfve',
    ).execute()
    data = res["items"]
    links = list(map(lambda el: el["link"], data))
    filtered_list = list(filter(lambda cn: cn.endswith(".pdf"), links))
    print(filtered_list)
    return res
    # pprint.pprint(res)

#
# def get_links_from_google(request_row):
#     response_data = search_result(request_row)
#     result = response_data["items"]
#     print(result)

#
# if __name__ == '__main__':
#     search_result()
