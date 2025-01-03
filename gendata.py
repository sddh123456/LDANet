def process_file(input_file, output_file, base_path, score):
    with open(input_file, 'r') as infile, open(output_file, 'a') as outfile:
        for line in infile:
            filename = line.strip()
            new_line = f"{base_path}{filename}.jpg,{score}\n"
            outfile.write(new_line)

def main():
    daytime_file = '/home/shengd/Data/bdd100k/images/100k/trainlist_daytime.txt'
    nighttime_file = './trainlist_nighttime.txt'
    output_file = 'data.txt'
    base_path = '/home/shengd/Data/bdd100k/images/100k/'

    process_file(daytime_file, output_file, base_path, 1)
    process_file(nighttime_file, output_file, base_path, 0)

if __name__ == "__main__":
    main()