import json

# with open('autotrader_cache.json') as f:
#     data = json.load(f)
#
# out = {}
#
# for k, v in data.items():
#     make = v['make']
#     model = v['model']
#     if make not in out:
#         out[make] = {}
#
#     if model not in out[make]:
#         out[make][model] = []
#
#     out[make][model].append(v)
#
# with open('autotrader_cache2.json', 'w') as f:
#     json.dump(out, f, indent=2)

# with open('autotrader_cache2.json') as f:
#     data = json.load(f)
#
#     for make in data:
#         for model in data[make]:
#             for data_dict in data[make][model]:
#                 if type(data_dict['year']) is str:
#                     tokens = data_dict['year'].split(' ', maxsplit=1)
#                     try:
#                         data_dict['year'] = int(tokens[0])
#                     except:
#                         print(make, model, data_dict['year'])
#
#     with open('autotrader_cache3.json', 'w') as f:
#         json.dump(data, f, indent=2)

with open('autotrader_cache3.json') as f:
    data = json.load(f)

    new_data = {}

    for make in data:

        if make.lower() not in new_data:
            new_data[make.lower()] = {}

        for model in data[make]:

            if model.lower() not in new_data[make.lower()]:
                new_data[make.lower()][model.lower()] = []

            for data_dict in data[make][model]:

                # get the hashes that exist in the list so far
                hashes = set(x['hash'] for x in new_data[make.lower()][model.lower()])

                if data_dict['hash'] not in hashes:
                    new_data[make.lower()][model.lower()].append(data_dict)
                    print('Adding', make, model, data_dict['hash'])
                else:
                    print('Duplicate hash', make, model, data_dict['hash'])


    with open('autotrader_cache4.json', 'w') as f2:
        json.dump(new_data, f2, indent=2)