import timeit
import os
from glob import glob
import numpy as np
import sys
from skimage import io
import psutil
from joblib import Parallel, delayed
from tqdm import tqdm
import scipy.signal
import math
#
def analyze_image_stack (path, vox_dim, matrix=None, air=None, res=None, z_corr=None):
    start = timeit.default_timer() #Start timing the function
    np.set_printoptions(threshold=sys.maxsize, linewidth=200)
    path0 = os.getcwd ()
    path_img = ("".join ([path, "\\img"])) #Path to the folder where the images are stores
    path_mask = ("".join ([path, "\\msk"])) #Path to the folder where the binary images of the fractures are stores
    path_store = ("".join ([path, "\\Minima_Analysis"])) #Path to the folder where the program store the data
    os.chdir (path_mask) #Change path to the folder where the mask_images are located
    mask_stack = sorted (glob("*.tif")) #List of the names of the TIF files in the folder (for the mask)
    if len(mask_stack)==0:
        raise ValueError("No TIF files found in the folder msk")
    os.chdir (path_img) #Change path to the folder where the images are located
    img_stack = sorted (glob("*.tif")) #List of the names of the TIFF files in the folder (for the original image)
    for i in range (0, len(mask_stack), 1): #List of the names of the TIFF files for the mask (including the path)
        mask_stack[i] = ("".join ([path_mask, "\\", mask_stack[i]]))
    if len(mask_stack)==0:
        raise ValueError("No TIF files found in the folder img")
    print ("Image shape is\tz: {}\ty: {}\tx: {}\n".format(len(img_stack), io.imread(img_stack[0], plugin='pil').shape[0], io.imread(img_stack[0], plugin='pil').shape[1]))
    os.chdir(path)
    if os.path.exists (path_store): #Check if the folder Minima_Analysis exists and has file inside (they will be removed).
        if len(os.listdir(path_store))>0:
            for f in os.listdir(path_store):
                os.remove(os.path.join(path_store, f))
    else:
        os.makedirs ("Minima_Analysis") #Create a folder to store the data inside the original image folder
    img3D = [] #An empty list for appending the images while opening slices
    img_mask3D = [] #An empty list for appending the mask_images while opening slices
    #Check the module to use for opening the images
    os.chdir (path_img)
    #Tries different modules of io.imread to open the images with correct shape
    if np.array(io.imread(img_stack[0], plugin='pil')).ndim==2:
        module_img='pil'
    elif np.array(io.imread(img_stack[0], plugin='imageio')).ndim==2:
        module_img='imageio'
    elif np.array(io.imread(img_stack[0], plugin='tifffile')).ndim==2:
        module_img='tifffile'
    elif np.array(io.imread(img_stack[0], plugin='imread')).ndim==2:
        module_img='imread'
    else:
        raise ValueError("Cannot open image properly with skimage.io")
    if np.array(io.imread (mask_stack[0], plugin='pil')).ndim==2:
        module_msk='pil'
    elif np.array(io.imread (mask_stack[0], plugin='imageio')).ndim==2:
        module_msk='imageio'
    elif np.array(io.imread(mask_stack[0], plugin='tifffile')).ndim==2:
        module_msk='tifffile'
    elif np.array(io.imread(mask_stack[0], plugin='imread')).ndim==2:
        module_msk='imread'
    else:
        raise ValueError("Cannot open mask properly with skimage.io")
    #Open the Images
    for i in zip (img_stack, mask_stack):
        img3D.append (io.imread (i[0], plugin=module_img)) #Open the 2D image slice as an array
        img_mask3D.append (io.imread (i[1], plugin=module_msk)) #Open the 2D mask_image slice as an array
    del img_stack; del mask_stack #Delete useless variables to empty the RAM
    img3D = np.array (img3D, copy=False) #Convert list to a numpy array
    img_mask3D = np.array (img_mask3D, copy=False, dtype="uint8") #Convert list to a numpy array
    img3D[:,0,:] = 0; img3D[:,-1,:] = 0; img3D[:,:,0] = 0; img3D[:,:,-1] = 0 #The grayscale image must have a 0 border
    img3D_rot =  np.rot90 (img3D, k=1, axes=(2,1)) #Rotate the 3D image for V analysis, step 1
    img3D_rot = img3D_rot[:,:,::-1] #Rotate the 3D image for V analysis, step 2
    # Analysis minima #######################################################################################################
    arr_minH3D, dataH = analyze_profile (img3D, img_mask3D, matrix, air, res, md='h')
    img_mask3D_rot = img_mask3D.copy()
    img_mask3D_rot[np.where(arr_minH3D>0)] = 0 #Avoid re-measuring same FTPs
    img_mask3D_rot =  np.rot90 (img_mask3D_rot, k=1, axes=(2,1)) #Rotate the 3D mask_image for vertical analysis, step 1
    img_mask3D_rot = img_mask3D_rot[:,:,::-1] #Rotate the 3D mask_image for vertical analysis, step 2
    arr_minV3D_rot, dataV = analyze_profile (img3D_rot, img_mask3D_rot, matrix, air, res, md='v')
    del img3D; del img_mask3D; del img3D_rot; del img_mask3D_rot #Delete useless variables to empty the RAM
    # H, V, and HV minima images ###########################################################################################
    os.chdir (path_store) #Change directory in order to store all data
    arr_minH3D = np.array (arr_minH3D, dtype="uint8", copy=False) #Convert the list of minima_images to a 3D array
    arr_minV3D_rot = np.array (arr_minV3D_rot, dtype="uint8", copy=False) #Convert the list of minima_images to a 3D array
    arr_minV3D_rot = arr_minV3D_rot[:,:,::-1] #Rotate the 3D image to the original position, step 1
    arr_minV3D_rot = np.rot90 (arr_minV3D_rot, k=1, axes=(-2,-1)) #Rotate the 3D image to the original position, step 2
    arr_minHV3D = np.add (arr_minH3D, arr_minV3D_rot, dtype="uint8") #Add the two 3D images
    del arr_minV3D_rot; del arr_minH3D #Delete useless variables to empty the RAM
    # Pad HV_min and HV_rec with zeroes to avoid problem when cropping #####################################################
    dataH = np.array(dataH, copy=False)#.astype (int)
    dataV = np.array(dataV, copy=False)#.astype (int)
    eg_eg = vox_dim//2
    dataH[:,:3]=dataH[:,:3]+eg_eg; dataH = list(dataH)
    dataV[:,:3]=dataV[:,:3]+eg_eg; dataV = list(dataV)
    arr_minHV3D = np.pad(arr_minHV3D, ((eg_eg,eg_eg), (eg_eg,eg_eg), (eg_eg, eg_eg)), mode='constant', constant_values=0)
    # Orientation Analysis ##################################################################################################
    elapsed_1 = timeit.default_timer(); print ("Minima analysis took {}s".format(round(timeit.default_timer() - start), 2))
    print ("\nCalculating orientations on {} points...".format(len(dataH)+len(dataV)))  
    if psutil.virtual_memory()[2]>85: raise ValueError("Low memory. {}% used.".format(psutil.virtual_memory()[2])) #Stops script if RAM is full
    half_dim = vox_dim//2
    rs = Parallel(n_jobs=6)(delayed(ftp_orientation_cpu)(arr_minHV3D, dataH[n], vox_dim, half_dim, H=True, core=True, z_c=z_corr) for n in tqdm(range(len(dataH)), desc='Orientation analysis', ncols=100)) #CPU1
    dataH = Parallel(n_jobs=6)(delayed(np.insert)(dataH[n], 3, rs[n]) for n in tqdm(range(len(dataH)), desc='Inserting orientation data', ncols=100)); del rs #CPU2
    elapsed_2 = timeit.default_timer(); print ("{}\tH points are done....it took {}s".format(len(dataH), round(elapsed_2 - elapsed_1), 2)) #Stops script if RAM is full
    if psutil.virtual_memory()[2]>85: raise ValueError("Low memory. {}% used.".format(psutil.virtual_memory()[2]))
    rs = Parallel(n_jobs=6)(delayed(ftp_orientation_cpu)(arr_minHV3D, dataV[n], vox_dim, half_dim, H=False, core=True, z_c=z_corr) for n in tqdm(range(len(dataV)), desc='Orientation analysis', ncols=100)) #CPU1
    dataV = Parallel(n_jobs=6)(delayed(np.insert)(dataV[n][3:], 0, ([dataV[n][0], dataV[n][2], dataV[n][1], rs[n][0], rs[n][1]])) for n in tqdm(range(len(dataV)), desc='Inserting orientation data', ncols=100)); del rs  #CPU2
    elapsed_3 = timeit.default_timer(); print ("{}\tV points are done...it took {}s".format(len(dataV), round(elapsed_3 - elapsed_2), 2))
    ##########################################################################################################################################
    arr_minHV3D = arr_minHV3D[eg_eg:-eg_eg,eg_eg:-eg_eg,eg_eg:-eg_eg] #Crop the image to remove the zero padded before and restore shape
    io.imsave ("img_3dHV.tif", arr_minHV3D); #Save the 3d minima image as TIF
    del arr_minHV3D; #del arr_HV3D_rec #Delete useless variables to empty the RAM
    print ("\nSaving data...")
    data_format = " z,   y,   x,  da,  dd, val, res,FWHM,  PH,  MA"
    dataH = np.array (dataH, copy=False)#.astype (int);
    dataH[:,:3]=dataH[:,:3]-eg_eg #Return correct z,y,x values after np.pad
    np.savetxt ("mfps_H.txt", dataH, fmt='%04d', delimiter=',', header=data_format)
    dataV = np.array (sorted (dataV, key=lambda x: (x[0], x[1], x[2])), copy=False)#.astype (int)
    dataV[:,:3]=dataV[:,:3]-eg_eg #Return correct z,y,x values after np.pad
    np.savetxt ("mfps_V.txt", dataV, fmt='%04d', delimiter=',', header=data_format)
    dataHV = np.concatenate ((dataH[:,:5], dataV[:,:5]), axis=0) #Concatenate array (only orientations)
    dataHV.view("i,i,i,i,i").sort(order=['f0','f1','f2'], axis=0)
    np.savetxt ("mfps_HV.txt", dataHV, fmt='%04d', delimiter=',', header=data_format[:22])
    print ("\nData H shape:\t{}\nData V shape:\t{}\nData H&V shape:\t{}\n".format(dataH.shape, dataV.shape, dataHV.shape))
    print ("\nThe function took {} seconds.\nEND".format(round(timeit.default_timer() - start),2))
    del dataH; del dataV; del dataHV
    os.chdir (path0)
#
def analyze_profile (arr3d, msk, mtrx, air, rs, md=None):
    """Analyze a 1D profile and return:
    - a binary array with ones at valleys positions;
    - an array that contains for each valley these measurements: valley_index, value_valley, LSF, FWHM, PH, MA"""
    arr_min3D = np.zeros((arr3d.shape)) #3D array of minima from H analysis
    data = []
    for n in tqdm(range(0, len(arr3d), 1), desc='Traverse Analysis', ncols=100):
        if psutil.virtual_memory()[2]>90: #If the RAM is full over the 85%, the program will stop
            raise ValueError("Low memory. {}% used.".format(psutil.virtual_memory()[2]))
        for m in range(0, len(arr3d[n])-1, 1):
            if np.count_nonzero(msk[n, m, :])>0:
                arr = arr3d[n, m, :].copy()
                msk_indx = np.nonzero (msk[n, m, :]) #Return the indices to keep (of value 1)
                peaks, _ = scipy.signal.find_peaks (arr) #Return peaks indexes of the array
                arr_v = -1*arr #In order to get valleys (recorded as peaks) when applying scipy.signal.find_peaks
                height_min = np.amin(arr_v[msk_indx]) #Minimum value in the thresholded image
                height_max = (np.amax(arr_v[msk_indx]) if np.amax(arr_v[msk_indx])<0 else -1) #Maximum value in the inverted image
                valleys, _ = scipy.signal.find_peaks (arr_v, height=(height_min, height_max)) #Return peaks indexes of the arr_v (valleys of arr)
                valleys = np.intersect1d (valleys, msk_indx, assume_unique=True) #Remove indexes not present in the mask
                del arr_v; del msk_indx #Delete useless variables to empty the RAM
                true_valleys = [] #Valleys that miss an edge (because close to the background which is equal to 0)
                res = [] #average of the 10-90% of the x-distance between the valley-edge 
                PH = [] #PH measured from the highest edge
                FWHM = [] #FWHM at half of PH
                MA = [] #MA (sum of the y-distances) from the highest edge to arr[n]
                valleys = valleys[valleys<mtrx]
                fwhm = ((mtrx-air)/2)+air
                if valleys.size > 0 and peaks.size > 0: #Check if the arrays are not empty
                    for i in valleys:                
                        idx = np.searchsorted (peaks, i) #It tell the index where the valley should be inserted to maintain order
                        edge_sx = peaks[idx-1] #Index of the left peak related to the valley i
                        edge_dx = peaks[idx] #Index of the right peak related to the valley i
                        if arr[edge_sx-1]==0 or arr[edge_dx+1]==0 or arr[i-1]==0 or arr[i+1]==0: #Check for false valleys/peaks
                            pass
                        else:
                            true_valleys.append(i)
                            size = ((edge_dx-edge_sx)/2)*0.8
                            if size>rs and arr[i]<fwhm:
                                FWHM_sx_edge = i-closest_indx(np.flip(arr[edge_sx:i+1]), fwhm) #Index position of FWHM at it's left side
                                FWHM_dx_edge = i+closest_indx(arr[i:edge_dx+1], fwhm) #Index position of FWHM at it's right side
                                if arr[FWHM_dx_edge]<fwhm:
                                    if arr[FWHM_dx_edge+1] == arr[FWHM_dx_edge]:
                                        while arr[FWHM_dx_edge+1] == arr[FWHM_dx_edge]:
                                            FWHM_dx_edge = FWHM_dx_edge+1
                                    a = float(arr[FWHM_dx_edge+1] - arr[FWHM_dx_edge])
                                    a_f = float(fwhm - arr[FWHM_dx_edge])
                                    perc = (a_f/a)
                                    FWHM_dx_edge = round(FWHM_dx_edge + perc, 1)
                                else:
                                    a = float(arr[FWHM_dx_edge] - arr[FWHM_dx_edge-1])
                                    a_f = float(arr[FWHM_dx_edge] - fwhm)
                                    perc = (a_f/a)
                                    FWHM_dx_edge = round(FWHM_dx_edge - perc, 1)
                                #
                                if arr[FWHM_sx_edge]<fwhm:
                                    if arr[FWHM_sx_edge-1] == arr[FWHM_sx_edge]:
                                        while arr[FWHM_sx_edge-1] == arr[FWHM_sx_edge]:
                                            FWHM_sx_edge = FWHM_sx_edge-1
                                    a = float(arr[FWHM_sx_edge-1] - arr[FWHM_sx_edge])
                                    a_f = float(fwhm - arr[FWHM_sx_edge])
                                    perc = (a_f/a)
                                    FWHM_sx_edge = round(FWHM_sx_edge + perc, 1)
                                else:
                                    a = float(arr[FWHM_sx_edge] - arr[FWHM_sx_edge+1])
                                    a_f = float(arr[FWHM_sx_edge] - fwhm)
                                    perc = (a_f/a)
                                    FWHM_sx_edge = round(FWHM_sx_edge - perc, 1)
                            else:
                                FWHM_sx_edge = 0.0 #Index position of FWHM at it's left side
                                FWHM_dx_edge = 0.0 #Index position of FWHM at it's right side
                                pass
                            FWHM.append ((FWHM_dx_edge - FWHM_sx_edge)*10) #FWHM for calibraction
                            res.append (size) #Edge response
                            pk_aver = (float(arr[edge_sx]+arr[edge_dx])/2)
                            pk_aver = pk_aver if pk_aver>air else mtrx
                            edg_edg_MA = pk_aver - arr[edge_sx:edge_dx+1] #Array of vertical heights for each pixel (from the max edge value)
                            edg_edg_MA = edg_edg_MA[edg_edg_MA>0]
                            MA.append (np.sum(edg_edg_MA/(pk_aver-air))*10) #Relative peaks MA
                            ph = (float(mtrx-arr[i])/float(mtrx-air))*1000 #PH estimated with the original formula
                            PH.append(round(ph, 1))
                if true_valleys:
                    value = np.take (arr3d[n, m, :], true_valleys) #Return the value of each valley
                    arr_min3D[n, m, true_valleys] = 255 #Fill the array of 255 in the valley index
                    z = [n]*len(true_valleys)
                    y = [m]*len(true_valleys)
                    data.append (zip (z, y, true_valleys, value, res, FWHM, PH, MA)) #Put the data together in a numpy array
    flatten = [k for i in data for k in i]
    flatten = np.array(flatten); print ('')
    return arr_min3D, flatten
#
def closest_indx (x1, x2):
    """Return from the indexes in x1, the closest to x2"""
    np.seterr(all='ignore')
    abs_=[]
    for i in x1:
        abs_.append(abs(i-x2))
    indx = abs_.index(min(abs_))
    return indx
#
def ftp_orientation_cpu(img, dt, d, h, H=True, core=True, z_c=None):
    """It measure the orientation of the plane that best fits the FTPs"""
    np.seterr(invalid='ignore');
    Z=dt[0]
    Y = dt[1] if H is True else dt[2]
    X = dt[2] if H is True else dt[1]
    arr = img[int(Z)-h:int(Z)+(d-h), int(Y)-h:int(Y)+(d-h), int(X)-h:int(X)+(d-h)].copy()
    orient=[]
    if np.count_nonzero(arr)<3: #Check if there are less than 3 points
        orient.append([-1.,-1.])
        pass
    else:
        if core is True:
            arr = np.rot90(arr, axes=(0,1)) #Core oriented
            indx = np.nonzero(arr) #Indexes of the non-zero points
        else:
            indx = np.nonzero(arr) #Indexes of the non-zero points
        z = indx[0]
        y = indx[1]
        x = indx[2]
        if z_c is not None and core is True:
            y=y*z_c
        elif z_c is not None and core is not True:
            z=z*z_c
        x = x - np.mean(x)
        y = y - np.mean(y);
        z = z - np.mean (z); #To center the cloud of points
        evals, evecs = np.linalg.eig(np.cov([x, y, z])) #Calculate eigenvalues and eigenvectors of the covariance matrix
        sort_indices = np.argsort(evals)[::-1]
        x_v3, y_v3, z_v3 = evecs[:, sort_indices[2]]
        north = np.asarray ([0,0,1]) #North => used to measure the dip direction
        zenith = np.asarray ([0,-1,0]) #Zenith => used to measure the dip angle
        normal = np.asarray ([x_v3, y_v3, z_v3]) #Normal of the fitting plane
        normal = normal*(-1.0 if normal[1]>=0 else 1.0) #Fix the normal if pointing in the opposite direction
        normalh = np.asarray ([normal[0], 0.0, normal[2]]) #Horizontal projection of the normal vector
        #Calculate the angle between vectors
        v1_u = normal/np.linalg.norm(normal) #For the dip_angle
        v2_u = zenith/np.linalg.norm(zenith) #For the dip_angle
        v3_u = normalh/np.linalg.norm(normalh) #For the dip_direction
        v4_u = north/np.linalg.norm(north) #For the dip_direction
        da = round(np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))),0)
        dd = round(np.degrees(np.arccos(np.clip(np.dot(v3_u, v4_u), -1.0, 1.0))),0) #dip_direction
        if normalh[0]<0 and dd<=180: #Correction if it measures the complementary angle for the dip_direction
            dd = 360-dd
        elif normalh[0]>0 and dd>180: #Correction if it measures the complementary angle for the dip_direction
            dd = 360-dd
        else:
            pass
        if math.isnan(da) or math.isnan(dd):
            orient.append([-1.,-1.])
        else:
            orient.append([da, dd])
    return orient[0]
#
path = (r"C:\Users\Franco\Desktop\py3\test_py3")
analyze_image_stack (path, vox_dim=11, matrix=255, air=0, res=1, z_corr=4)