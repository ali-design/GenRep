from os import listdir
from os.path import isfile, join, basename, isfile
import numpy as np

def make_html(home_dir):
    curr_dir = home_dir.split('/')[-1]
    std_list = ['iso-nbr-1', 'iso-nbr-2', 'iso-nbr-3', 'iso-nbr-4', 'iso-nbr-5', 'iso-nbr-6', 'steer-nbr-1','steer-nbr-2','steer-nbr-3','steer-nbr-4','steer-nbr-5','steer-nbr-6']
    try:
        index_html_path = join(home_dir, "index.html")
        print('Saving index file in', index_html_path)
        fid = open(index_html_path, 'w', encoding = 'utf-8')

        fid.write('<table style="text-align:center;">')
        header_str = '<tr><td>anchore filename</td><td>anchore</td>'
        for i in range(len(std_list)):
            if i == 6:
                header_str += '<td style="width:100px"> </td>'
            header_str += '<td>'+std_list[i]+'</td>'
        header_str += '</tr>'
        fid.write(header_str)
        home_dir = join(home_dir, 'train')
        dir_list = listdir(home_dir)   
        for ii in range(len(dir_list)): 
        # for ii in range(10):
            images_dir = join(home_dir, dir_list[ii])
            if isfile(images_dir):
                continue
            file_names = [f for f in listdir(images_dir) if join(images_dir, f).endswith('_anchor.png')]

            for i in range(len(file_names)):
                fid.write('<tr>')
                anchor_file_name = join('train', basename(images_dir), file_names[i])
                fid.write('<td>' + anchor_file_name + '</td>')

                anchor_file_name = join('train', basename(images_dir), file_names[i])
                fid.write('<td><a href="' + anchor_file_name + '"><img style="width:128px;height:128px;" src="' +
                        anchor_file_name + '"/></a></td>')
                for j in range(len(std_list)):
                    if std_list[j] == 'iso-nbr-1':
                        std_file_name = anchor_file_name.replace('anchor', 'iso_1')
                    elif std_list[j] == 'iso-nbr-2':
                        std_file_name = anchor_file_name.replace('anchor', 'iso_2')
                    elif std_list[j] == 'iso-nbr-3':
                        std_file_name = anchor_file_name.replace('anchor', 'iso_3')
                    elif std_list[j] == 'iso-nbr-4':
                        std_file_name = anchor_file_name.replace('anchor', 'iso_4')
                    elif std_list[j] == 'iso-nbr-5':
                        std_file_name = anchor_file_name.replace('anchor', 'iso_5')
                    elif std_list[j] == 'iso-nbr-6':
                        std_file_name = anchor_file_name.replace('anchor', 'iso_6')

                    elif std_list[j] == 'steer-nbr-1':
                        fid.write('<td style="width:100px"></td>')
                        std_file_name = anchor_file_name.replace('anchor', 'steer_1')
                    elif std_list[j] == 'steer-nbr-2':
                        std_file_name = anchor_file_name.replace('anchor', 'steer_2')
                    elif std_list[j] == 'steer-nbr-3':
                        std_file_name = anchor_file_name.replace('anchor', 'steer_3')
                    elif std_list[j] == 'steer-nbr-4':
                        std_file_name = anchor_file_name.replace('anchor', 'steer_4')
                    elif std_list[j] == 'steer-nbr-5':
                        std_file_name = anchor_file_name.replace('anchor', 'steer_5')
                    elif std_list[j] == 'steer-nbr-6':
                        std_file_name = anchor_file_name.replace('anchor', 'steer_6')
                        
                    fid.write('<td style="padding:0px"><a href="' + std_file_name + '"><img style="width:128px;height:128px;" src="' +
                        std_file_name + '"/></a></td>')
                fid.write('</tr>')
        fid.write('</table>')

    finally:
        fid.close()

make_html('/data/vision/phillipi/ganclr/datasets/biggan256tr1-png_paper_figure_10_samples')
