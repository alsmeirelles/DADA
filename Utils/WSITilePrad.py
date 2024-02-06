import numpy as np
import openslide
import sys
import os
import time
from PIL import Image
import argparse
import multiprocessing

#Local imports
#from import_module import import_parents

#if __name__ == '__main__' and __package__ is None:
#    import_parents(level=1)
    
#Local functions
#from Preprocessing import background,white_ratio


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
        

def get_patch_label(hm,imid,x,y,pw,hms,debug=False,cdict=None):
    """
    Extracts patch labels from heatmaps
    """
    hm_x = round(round(x*hms[imid]['mpp'])/50)
    hm_y = round(round(y*hms[imid]['mpp'])/50)
    hm_w = round(round(pw*hms[imid]['mpp'])/50)
    #hm_x = round((x+pw-1)/pw)
    #hm_y = round((y+pw-1)/pw)
    #hm_w = round((2*pw-1)/pw)
    
    if not cdict is None:
        cdict.setdefault((hm_x,hm_y),[])
        cdict[(hm_x,hm_y)].append((x,y))
        
    if not 'cancer_t' in hms[imid]:
        cancer_t = os.path.split(os.path.dirname(hms[imid]['path']))[-1]
        hms[imid]['cancer_t'] = cancer_t
    hm_roi = hm.crop((hm_x,hm_y,hm_x+hm_w,hm_y+hm_w))
    np_hm = np.array(hm_roi)
    if debug:
        hm_roi.show()
        time.sleep(5)

    if len(np_hm.shape) < 3:
        print("Error processing WSI ({}), patch shape: {}".format(imid,np_hm.shape))
        print("Crop region: {}".format((hm_x,hm_y,hm_x+hm_w,hm_y+hm_w)))

    r,g,b = np_hm[:,:,0],np_hm[:,:,1],np_hm[:,:,2]
    r_count = np.where(r == 255)[0].shape[0]
    b_count = np.where(b == 255)[0].shape[0]

    if debug:
        print("Red pixels: {}".format(r_count))
        print("Blue pixels: {}".format(b_count))

    if b_count > 0 and r_count/b_count > 0.2:
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
            if img_id in hms:
                print("Duplicate heatmap name ({}):\n - {}\n - {}".format(img_id,hms[img_id],os.path.join(hm_path,k,img)))
            hms[img_id] = {'path':os.path.join(hm_path,k,img)}

    removed = 0
    final = []
    for w in wsis:
        w_id = w.split('.')[0]
        if not w_id in hms:
            print("Slide {} has no heatmap.".format(w))
            removed += 1
        else:
            final.append(w)

    if removed == 0:
        print("All WSIs have a corresponding heatmap.")
        
    return final,hms

def make_tiles(slide,output_folder,patch_size,wr,csv_dir,debug=False):
    """
    Ask for a tile of size patch_size_20X, but output will only have this size if it is
    a value divisible by 100.
    """
    slide_name = os.path.basename(slide)
    imid = slide_name.split(".")[0]

    csv = os.path.join(csv_dir,"-".join(["prediction",imid]))
    if not os.path.isfile(csv):
        print("No CSV file for WSI: {}".format(slide_name))
        return 0,0

    if debug:
        print("Starting SLIDE: {}".format(slide_name))

    to_dir = os.path.join(output_folder,imid)
    if not os.path.isdir(to_dir):
        os.mkdir(to_dir)        

    csv_data = None
    with open(csv,'r') as fd:
        csv_data = fd.readlines()
        csv_data.pop(0)
        
    try:
        oslide = openslide.OpenSlide(slide);
    except Exception as e:
        print('{}: exception caught'.format(slide));
        print(e)
        sys.exit(1);

    width = oslide.dimensions[0];
    height = oslide.dimensions[1];

    pcount = 0
    pos_count = 0
    
    #Start patch extraction
    print(slide_name, width, height);
        
    for d in csv_data:
        x,y,p1,p2,p3 = d.split(' ')
        x,y = int(x),int(y)
        x = round(x - patch_size/2)
        x = x if x >= 0 else 0
        y = round(y - patch_size/2)
        y = y if y >= 0 else 0
        p1,p2,p3 = float(p1),float(p2),float(p3)
        if (p1+p2+p3) == 0.0:
            continue
        
        patch = oslide.read_region((x, y), 0, (patch_size, patch_size));
        np_patch = np.array(patch)
        #if not background(np_patch) and white_ratio(np_patch) <= wr:
        if p2 >= 0.5:
            pos_count += 1
            label = 1
        else:
            label = 0

        fname = '{}/{}/{}-{}-{}-{}-{}_{}.png'.format(output_folder,imid, imid, x, y, patch_size, np_patch.shape[0],label)
            
        if debug:
            print("    - {}: prob {}".format(os.path.basename(fname),p2))

        if not patch.info.get('icc_profile') is None:
            del(patch.info['icc_profile'])
        
        patch.save(fname);
        pcount += 1

        del(np_patch)
            
    oslide.close()
    if debug:
        print("Finished slide {}: {} total positives".format(imid,pos_count))
        
    sys.stdout.flush()
    return pcount,pos_count

def generate_label_files(tdir):
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
                    fields = [im,label,'-'.join(fields[:6]),fields[6],fields[7]]
                    fd.write("{}\n".format(" ".join(fields)))

    print("Done generating label files")
    
if __name__ == "__main__":

    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Extract interest patches from a WSI \
        discarding background.')
        
    parser.add_argument('-ds', dest='ds', type=str,default='WSI', 
        help='Path to WSIs to tile (directory containing .svs images).')
    parser.add_argument('-rf', dest='rf', type=str,default='model-outputs', 
        help='Path to files defining regions of interest (directory containing .csv files).')    
    parser.add_argument('-nwsi', dest='nwsi', type=int, 
        help='Tile this many WSIs (Default: all)', default=0,required=False)    
    parser.add_argument('-od', dest='out_dir', type=str, default='Patches', 
        help='Save extracted patches to this location.')
    parser.add_argument('-mp', dest='mp', type=int, 
        help='Use multiprocessing. Number of processes to spawn', default=2,required=False)
    parser.add_argument('-label', action='store_true', dest='label',
        help='Generate labels for the patches from heatmaps.',default=False)
    parser.add_argument('-txt_label', action='store_true', dest='txt_label',
        help='Generate text file labels for the patches from heatmaps.',default=False)    
    parser.add_argument('-ps', dest='patch_size', type=int, 
        help='Patch size in 20x magnification (Default is 350 px)', default=350,required=False)
    parser.add_argument('-wr', dest='white', type=float, 
        help='Maximum white ratio allowed for each patch (Default: 0.20)', default=0.2,required=False)
    parser.add_argument('-db', action='store_true', dest='debug',
        help='Use to make extra checks on labels and conversions.',default=False)

    
    config, unparsed = parser.parse_known_args()

    if not os.path.exists(config.out_dir):
        os.mkdir(config.out_dir);

    if not (os.path.isdir(config.ds) or os.path.isdir(config.rf)):
        print("Path not found: check {} and {}".format(config.ds,config.rf))
        sys.exit(1)

    if config.txt_label:
        config.label = True
        
    wsis = os.listdir(config.ds)
    wsis = list(filter(lambda x:x.split('.')[-1] == 'svs',wsis))

    if config.nwsi > 0:
        wsis = wsis[:config.nwsi]
    
    results = None
    total_patches = 0
    total_pos = 0
        
    with multiprocessing.Pool(processes=config.mp) as pool:
        results = [pool.apply_async(make_tiles,(os.path.join(config.ds,i),config.out_dir,config.patch_size,
                                                    config.white,config.rf,config.debug)) for i in wsis]
        total_patches = sum([r.get()[0] for r in results])
        total_pos = sum([r.get()[1] for r in results])

    print("Total of patches generated: {}".format(total_patches))
    print("Total of positive patches generated: {} ({:2.2f} %)".format(total_pos,100*total_pos/total_patches))

    if config.txt_label:
        generate_label_files(config.out_dir)
