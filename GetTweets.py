import requests, json, sys

item_dict = {}


def get_tweets(tweetId):
    headers = {
        'Authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAAH7lOgEAAAAARCzkM9MgkvTEaigYhjDVvvG8FdA'
                         '%3DSvk4rFHgObNRSoRK9TX2NOeowyDM595ZoDzJFF3RySQ3eKL242',
    }
    response = requests.get('https://api.twitter.com/2/tweets/' + str(tweetId), headers=headers)
    if response.status_code == 429:
        quit()
    json_data = json.loads(response.text)
    d_list = []
    for key1, value1 in json_data.items():
        if key1 == 'errors':
            continue
        else:
            for key2, value2 in value1.items():
                d_list.append(value2)
    return d_list


try:
    for line in open(sys.argv[1]):
        fields = line.rstrip().split('\t')
        tweetid = fields[0]
        userid = fields[1]
        abuseid = fields[2]
        tweet = None
        text = "Not Available"
        print(userid)
        if tweetid in item_dict:
            continue
        else:
            try:
                item_dict[tweetid] = []
                item_dict[tweetid].append(get_tweets(tweetid)[1])
                item_dict[tweetid].append(abuseid)
                print(item_dict[tweetid])

            except Exception:
                continue


except IndexError:
    print('Incorrect arguments specified (may be you didn\'t specify any arguments..')
    print('Format: python [scriptname] [inputfilename] > [outputfilename]')

with open(sys.argv[2], "w", encoding="utf-8") as f:
    f.write("{\n")
    for k in item_dict.keys():
        if item_dict[k]:
            print(k, item_dict[k])
            f.write("'{}':'{}'\n".format(k, item_dict[k]))
f.close()


