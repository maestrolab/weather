import csv

# Ranges indicate the first and last profile data was collected from
latitude_range = [55, 58]
longitude_range = [-144, -53]

profiles_tuple = (latitude_range[0], longitude_range[0],
                  latitude_range[1], longitude_range[1])

'''PUT CHECK FOR EXISTING FILE SO THAT IT DOESN'T GET OVERWRITTEN'''
ADD CHECK FOR EXISTING FILE AND THEN DELETE THIS MESSAGE (OR JUST COMMENT ME OUT IF YOU KNOW WHAT YOU ARE DOING)

# txt_file = r"txt_files/error_[%i,%i]_[%i,%i].txt" % profiles_tuple
# csv_file = r"csv_files/csv_[%i,%i]_[%i,%i].csv" % profiles_tuple

in_txt = csv.reader(open(txt_file, "r"), delimiter = '\t')
out_csv = csv.writer(open(csv_file, 'w'))

out_csv.writerows(in_txt)
