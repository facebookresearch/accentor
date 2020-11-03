import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="./MultiWOZ_2.1/data.json", type=str, required=False, help="Path to the MultiWOZ dataset.")
    args = parser.parse_args()
    
    with open("candidates-multiwoz.json", "r", encoding='utf8') as f:
        augmentation = json.load(f)

    with open(args.source, "r", encoding='utf8') as f:
        data = json.load(f)

    data = {x:data[x] for x in data if x in augmentation}

    for x in data:
        for i in range(1, len(data[x]["log"]), 2):
            data[x]["log"][i]["beginning"] = []
            data[x]["log"][i]["end"] = []
        for cc in augmentation[x]:
            data[x]["log"][cc[0]][cc[1]] += [{"candidate": cc[2], "label": cc[3], "justification": cc[4]}]

    with open("accentor-multiwoz-1k.json", "w", encoding='utf8') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
