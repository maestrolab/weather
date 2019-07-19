latitudes = [[32,32],[32,32]]
longitudes = [[-144,-128],[-127,-53]]

new_profile_tuple = (latitudes[0][0], longitudes[0][0],
                     latitudes[1][1], longitudes[1][1])

profiles_tuple = ((latitudes[0][0], longitudes[0][0],
                  latitudes[0][1], longitudes[0][1]),
                  (latitudes[1][0], longitudes[1][0],
                  latitudes[1][1], longitudes[1][1]))

append_to = 'txt_files/error_[%i,%i]_[%i,%i].txt' % profiles_tuple[0]
append_from = 'txt_files/error_[%i,%i]_[%i,%i].txt' % profiles_tuple[1]
new_file = 'txt_files/error_[%i,%i]_[%i,%i].txt' % new_profile_tuple

ADD CHECK FOR EXISTING FILE AND THEN DELETE THIS MESSAGE (OR JUST COMMENT ME OUT IF YOU KNOW WHAT YOU ARE DOING)

f = open(append_to, 'r')
f_lines = f.readlines()
f.close()

g = open(append_from, 'r')
g_lines = g.readlines()
g.close()

h = open(new_file,'w')
for line in f_lines:
    h.write(line)
for line in g_lines:
    h.write(line)
h.close()
