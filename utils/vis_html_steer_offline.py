from os import listdir
from os.path import isfile, join, basename, isfile
import numpy as np

def make_html(home_dir):
    std_list = ['zoom', 'color', 'rot3d']
    try:
        index_html_path = join(home_dir, "index.html")
        print('Saving index file in', index_html_path)
        fid = open(index_html_path, 'w', encoding = 'utf-8')

        fid.write('<table style="text-align:center;">')
        header_str = '<tr><td>anchore filename</td><td>anchore</td>'
        for i in range(len(std_list)):
            header_str += '<td>'+std_list[i]+'</td>'
        header_str += '</tr>'
        fid.write(header_str)
        home_dir = join(home_dir, 'biggan256tr1-png_steer_zoom_100/train')
        dir_list = listdir(home_dir)   
        for ii in range(len(dir_list)): 
        # for ii in range(10):
            images_dir = join(home_dir, dir_list[ii])
            if isfile(images_dir):
                continue
            file_names = [f for f in listdir(images_dir) if join(images_dir, f).endswith('_anchor.png')]

            # for i in range(len(file_names)):
            for i in range(3):
                fid.write('<tr>')
                anchor_file_name = join('/train', basename(images_dir), file_names[i])
                fid.write('<td>' + anchor_file_name + '</td>')
                anchor_file_name = join('biggan256tr1-png_steer_zoom_100/train', basename(images_dir), file_names[i])
                fid.write('<td><a href="' + anchor_file_name + '"><img src="' +
                        anchor_file_name + '"/></a></td>')
                for j in range(len(std_list)):
                    shuffle = np.random.uniform()
                    if shuffle < 0.5:
                        std_file_name = anchor_file_name.replace('anchor', 'palpha')
                    else:
                        std_file_name = anchor_file_name.replace('anchor', 'nalpha')
 
                    std_file_name = std_file_name.replace('zoom', std_list[j])
                    if std_list[j] == 'color':
                        if shuffle <= 0.25:
                            std_file_name = std_file_name.replace('sample', 'W_sample')
                        elif shuffle > 0.25 and shuffle <= 0.5:
                            std_file_name = std_file_name.replace('sample', 'R_sample')
                        elif shuffle > 0.5 and shuffle <= .75:
                            std_file_name = std_file_name.replace('sample', 'G_sample')
                        elif shuffle > 0.75:
                            std_file_name = std_file_name.replace('sample', 'B_sample')
                                                    
                        std_file_name = std_file_name.replace('nalpha', 'palpha')

                    fid.write('<td><a href="' + std_file_name + '"><img src="' +
                        std_file_name + '"/></a></td>')
                fid.write('</tr>')
        fid.write('</table>')

    finally:
        fid.close()

make_html('/data/scratch/jahanian/ganclr_results_2')
# make_html('/data/scratch-oc40/jahanian/ganclr_results/biggan256tr1-png_steer_rnd_100/val')
# make_html('./utils/val')