from os import listdir
from os.path import isfile, join, basename, isfile
import numpy as np

def make_html(home_dir):
    std_list = ['0.1', '0.2', '0.5', '1.0']
    try:
        index_html_path = join(home_dir, "index.html")
        print('Saving index file in', index_html_path)
        fid = open(index_html_path, 'w', encoding = 'utf-8')

        fid.write('<table style="text-align:center;">')
        header_str = '<tr><td>anchore filename</td><td>anchore</td>'
        for i in range(len(std_list)):
            header_str += '<td>std_'+std_list[i]+'</td>'
        header_str += '</tr>'
        fid.write(header_str)

        dir_list = listdir(home_dir)   
        # for ii in range(len(dir_list)): 
        for ii in range(3):
            images_dir = join(home_dir, dir_list[ii])
            if isfile(images_dir):
                continue
            file_names = [f for f in listdir(images_dir) if join(images_dir, f).endswith('_anchor.png')]

            for i in range(len(file_names)):
                fid.write('<tr>')
                anchor_file_name = join('.', basename(images_dir), file_names[i])
                fid.write('<td>' + anchor_file_name + '</td>')
                fid.write('<td><a href="' + anchor_file_name + '"><img src="' +
                        anchor_file_name + '"/></a></td>')
                for j in range(len(std_list)):
                    std_file_name = anchor_file_name.replace('anchor', std_list[j])
                    fid.write('<td><a href="' + std_file_name + '"><img src="' +
                        std_file_name + '"/></a></td>')
                fid.write('</tr>')
        fid.write('</table>')

    finally:
        fid.close()

make_html('/data/scratch-oc40/jahanian/ganclr_results/biggan256tr1-png_steer_rnd_100/val')
# make_html('./utils/val')