import numpy as np
import openslide
import sys
import os
import time
from PIL import Image
import argparse
import multiprocessing

#Local functions
from Preprocessing import background,white_ratio


def _check_heatmap_size(imid,width,height,hms):
    print("Heatmap: {}".format(hms[imid]['path']))
    hm = Image.open(hms[imid]['path'])
    hm_w,hm_h = hm.size

    pw = round(50*hm_w/hms[imid]['mpp'])
    ph = round(50*hm_h/hms[imid]['mpp'])
    if pw != width or ph != height:
        print("Shape mismatch:\n -WSI width:{0}; HM converted:{1};\n -WSI height:{2}; HM converted:{3}".format(width,pw,height,ph))
    else:
        print("Shape ok!")
        

def get_patch_label(hm,imid,x,y,pw,hms,debug=False):
    """
    Extracts patch labels from heatmaps
    """
    hm_x = x*hms[imid]['mpp']/50
    hm_y = y*hms[imid]['mpp']/50

    hm_w = pw*hms[imid]['mpp']/50
    if not 'cancer_t' in hms[imid]:
        cancer_t = os.path.split(os.path.dirname(hms[imid]['path']))[-1]
        hms[imid]['cancer_t'] = cancer_t
    hm_roi = hm.crop((hm_x,hm_y,hm_x+hm_w,hm_y+hm_w))
    np_hm = np.array(hm_roi)
    if debug:
        hm_roi.show()
        time.sleep(5)

    r,g,b = np_hm[:,:,0],np_hm[:,:,1],np_hm[:,:,2]
    r_count = np.where(r == 255)[0].shape[0]
    b_count = np.where(b == 255)[0].shape[0]

    if debug:
        print("Red pixels: {}".format(r_count))
        print("Blue pixels: {}".format(b_count))

    if r_count > 0:
        return 1
    else:
        return 0
    
def check_heatmaps(hm_path,wsis):
    """
    Checks if all WSIs in source dir have a corresponding heatmap
    
    Returns: dictionary of dictionarys
    k -> {'path': path to the slide's heatmap}
    """
    if hm_path is None or not os.path.isdir(hm_path):
        print("Path not found: {}".format(hm_path))
        sys.exit(1)
        
    cancer_t = os.listdir(hm_path)
    cancer_t = list(filter(lambda d: os.path.isdir(os.path.join(hm_path,d)),cancer_t))

    hms = {}
    for k in cancer_t:
        for img in os.listdir(os.path.join(hm_path,k)):
            img_id = img.split('.')[0]
            hms[img_id] = {'path':os.path.join(hm_path,k,img)}

    removed = 0
    for w in wsis:
        w_id = w.split('.')[0]
        if not w_id in hms:
            print("Slide {} has no heatmap.".format(w))
            wsis.remove(w)
            removed += 1

    if removed == 0:
        print("All WSIs have a corresponding heatmap.")
        
    return wsis,hms

def make_tiles(slide_name,output_folder,patch_size_20X,wr,hms,debug=False):
    imid = os.path.basename(slide_name).split('.')[0]
    hms[imid]['svs'] = os.path.basename(slide_name)
    
    try:
        oslide = openslide.OpenSlide(slide_name);
        #mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
        if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
            mpp = float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
            print("Image MPP: {}".format(mpp))
            if not hms is None:
                hms[imid]['mpp'] = mpp
            mag = 10.0 / mpp
        elif "XResolution" in oslide.properties:
            mag = 10.0 / float(oslide.properties["XResolution"]);
        elif 'tiff.XResolution' in oslide.properties:   # for Multiplex IHC WSIs, .tiff images
            mag = 10.0 / float(oslide.properties["tiff.XResolution"]);
        else:
            mag = 10.0 / float(0.254);
    except:
        print('{}: exception caught'.format(slide_name));
        exit(1);

    pw = int(patch_size_20X * mag / 20);
    width = oslide.dimensions[0];
    height = oslide.dimensions[1];

    pcount = 0
    pos_count = 0
    print(slide_name, width, height);
    if not hms is None and debug:
        _check_heatmap_size(imid,width,height,hms)
        
    for x in range(1, width, pw):
        for y in range(1, height, pw):
            if x + pw > width:
                #pw_x = width - x;
                continue
            else:
                pw_x = pw;
            if y + pw > height:
                #pw_y = height - y;
                continue
            else:
                pw_y = pw;

            if (int(patch_size_20X * pw_x / pw) <= 0) or \
              (int(patch_size_20X * pw_y / pw) <= 0) or \
              (pw_x <= 0) or (pw_y <= 0):
                continue;

            patch = oslide.read_region((x, y), 0, (pw_x, pw_y));
            np_patch = np.array(patch)
            if not background(np_patch) and white_ratio(np_patch) <= wr:
                patch = patch.resize((int(patch_size_20X * pw_x / pw), int(patch_size_20X * pw_y / pw)), Image.ANTIALIAS);
                if not hms is None:
                    label = get_patch_label(oslide,imid,x,y,pw,hms,debug)
                    if label > 0:
                        pos_count += 1
                    to_dir = os.path.join(output_folder,hms[imid]['cancer_t'])
                    if not os.path.isdir(os.path.join(to_dir)):
                        os.mkdir(to_dir)
                    fname = os.path.join(to_dir,'{}-{}-{}-{}-{}_{}.png'.format(imid,x, y, pw, patch_size_20X,label));
                else:
                    fname = '{}/{}-{}-{}-{}-{}.png'.format(output_folder, imid, x, y, pw, patch_size_20X);
                patch.save(fname);
                pcount += 1

            del(np_patch)

    oslide.close()
    return pcount,pos_count

def generate_label_files(tdir,hms):
    """
    CellRep datasources use label text files to store ground truth for each patch.
    File: label.txt
    Format: file_name label source_svs x y
    """

    for d in os.listdir(tdir):
        c_dir = os.path.join(tdir,d)
        if os.path.isdir(c_dir):
            patches = os.listdir(c_dir)
            patches = list(filter(lambda p: p.endswith('.png'),patches))
            with open(os.path.join(c_dir,'label.txt'),'w') as fd:
                for im in patches:
                    fields = im.split('.')[0].split('_')
                    label = fields[1]
                    fields = fields[0].split('-')
                    fields = [im,label,'-'.join(fields[:5]),fields[6],fields[7]]
                    fd.write("{}\n".format(" ".join(fields)))

    print("Done generating label files")
    
if __name__ == "__main__":

    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Extract tiles from a WSI \
        discarding background.')
        
    parser.add_argument('-ds', dest='ds', type=str,default='WSI', 
        help='Path to WSIs to tile (directory containing .svs images).')        
    parser.add_argument('-od', dest='out_dir', type=str, default='Patches', 
        help='Save extracted patches to this location.')
    parser.add_argument('-mp', dest='mp', type=int, 
        help='Use multiprocessing. Number of processes to spawn', default=2,required=False)
    parser.add_argument('-label', action='store_true', dest='label',
        help='Generate labels for the patches from heatmaps.',default=False)
    parser.add_argument('-txt_label', action='store_true', dest='txt_label',
        help='Generate labels for the patches from heatmaps.',default=False)    
    parser.add_argument('-hm', dest='heatmap', type=str,default=None, 
        help='Heatmaps path.')
    parser.add_argument('-ps', dest='patch_size', type=int, 
        help='Patch size in 20x magnification (Default 500)', default=500,required=False)
    parser.add_argument('-wr', dest='white', type=float, 
        help='Maximum white ratio allowed for each patch (Default: 0.20)', default=0.2,required=False)
    parser.add_argument('-db', action='store_true', dest='debug',
        help='Use to make extra checks on labels and conversions.',default=False)    
    
    config, unparsed = parser.parse_known_args()

    if not os.path.exists(config.out_dir):
        os.mkdir(config.out_dir);

    if not os.path.isdir(config.ds):
        print("Path not found: {}".format(config.ds))
        sys.exit(1)

    if config.txt_label:
        config.label = True
        
    wsis = os.listdir(config.ds)
    wsis = list(filter(lambda x:x.split('.')[-1] == 'svs',wsis))
    
    results = None
    total_patches = 0
    total_pos = 0
    
    hms = None
    if config.label:
        wsis,hms = check_heatmaps(config.heatmap,wsis)
        
    with multiprocessing.Pool(processes=config.mp) as pool:
        results = [pool.apply_async(make_tiles,(os.path.join(config.ds,i),config.out_dir,config.patch_size,
                                                    config.white,hms,config.debug)) for i in wsis]
        total_patches = sum([r.get()[0] for r in results])
        total_pos = sum([r.get()[1] for r in results])

    print("Total of patches generated: {}".format(total_patches))
    print("Total of positive patches generated: {} ({:2.2f} %)".format(total_pos,100*total_pos/total_patches))

    if config.txt_label:
        generate_label_files(config.out_dir,hms)
