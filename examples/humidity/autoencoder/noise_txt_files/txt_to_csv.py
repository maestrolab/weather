import csv

# Date that noise differences were calculated
date = '7_12_19' + '_0-100'

'''PUT CHECK FOR EXISTING FILE SO THAT IT DOESN'T GET OVERWRITTEN'''
# ADD CHECK FOR EXISTING FILE AND THEN DELETE THIS MESSAGE (OR JUST COMMENT ME OUT IF YOU KNOW WHAT YOU ARE DOING)

txt_file = r"%s.txt" % date
csv_file = r"%s.csv" % date

in_txt = csv.reader(open(txt_file, "r"), delimiter = '\t')
out_csv = csv.writer(open(csv_file, 'w'))

out_csv.writerows(in_txt)
