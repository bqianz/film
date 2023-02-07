import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import pickle
# from sarlab.gammax import readBin, writeBin
# from sarlab.tools.least_squares import lsq_fit
# from sarlab.tools.filters import smooth
from skimage.transform import rescale, resize
from sklearn.decomposition import FastICA
import glob
import os
import time
import re

# user-modifiable parameters
n_comp = 5 #10 # number of sources
fig_col_comp=3
fig_row_comp=2
udir = '/local-scratch/users/btrabus/landslides/fels/ica-testing/input_demod_unw/'
fdir = '/local-scratch/users/btrabus/landslides/fels/ica-testing/convfac_tsx/'
#ica_omit_list= ['20060723_20060816']  #only noisy and super long ifgs excluded
#
ica_omit_list= ['20030621_20030715','20040826_20040919','20050704_20050728','20050821_20050914','20060723_20060816','20060816_20060909','20060602_20060718','20150618_20150712','20150712_20150805','20140810_20140903','20140903_20140927'] 

# global settings
n_ifs = 52
flatten_order = 'C'
rg_samples = 9630
az_lines = 6477
dtype = 'f4'

whiten= True
iters= 5000
tolerance=1e-4
shuffle= True
center = False #True
scale_fact = 10

col_start = 0
row_start = 0
n_col = int(az_lines/scale_fact); 
n_row = int(rg_samples/scale_fact)
fig_col = 7
fig_row = 8
if_lim = (-10,10)
s_lim = (-0.005,0.005)

sensors=['tsx','rsat','ers','alos']
sensor_periods=np.array([11,24,35,46])
wave_ratios=np.array([3.1,5.7,5.7,23.6])/3.1


def extract_dates(filename):
	dates=np.fromstring(re.sub('[^0-9]',' ',filename),sep=' ',dtype='int')
	dates= dates[dates > 19000000].astype('str')
	return dates

# def date2jd_wrong(datestr):  # need to rerplace with the routine below everywhere !!!!
#     datestr=np.atleast_1d(datestr)
#     nd=np.size(datestr)
#     jd=np.zeros(nd)
#     for k in range(0,nd):
#         datestr_k=datestr[k]
#         yr=int(datestr_k[0:4])
#         mo=int(datestr_k[4:6])
#         dy=int(datestr_k[6:8])
#         a = int(yr/100)
#         b = int(a/4)
#         c = 2-a+b
#         e = int(365.25*(yr+4716))
#         f = int(30.6001*(mo+1))
#         jd[k]= c+dy+e+f-1524.5
#     if nd ==1:
#         jd=jd[0]
#     return(jd)

def date2jd(datestr):   #this conversion appears to be correct (original in gammax/my_py_code was wrong)
    datestr=np.atleast_1d(datestr)
    nd=np.size(datestr)
    jd=np.zeros(nd)
    for k in range(0,nd):
        datestr_k=datestr[k]
        yr=int(datestr_k[0:4])
        mo=int(datestr_k[4:6])
        dy=int(datestr_k[6:8])
        
        a = int((14-mo)/12)
        y = yr+4800-a
        m = mo + 12*a - 3
        jd[k]= (dy + int((153*m+2)/5) + y*365 + int(y/4) - 32083)
    if nd ==1:
        jd=jd[0]
    jd-=13   #Gregorian to Julian
    return(jd)   

def jd2date(jd):   #this conversion appears to be correct (original in gammax/my_py_code was wrong)
    jd=np.atleast_1d(jd)
    nd=np.size(jd)
    datestr=np.zeros(nd,dtype='U8')
    jd+=13    #Julian to Gregorian
    #jd+=1 #not sure why back conversion is off by one day
    for k in range(0,nd):
        c = int(jd[k]) + 32083
        d = int((4*(c+365))/1461) - 1
        e = c - int((1461*d)/4)
        m = int((5*(e-1)+2)/153)
        dy   = e - int((153*m+2)/5)
        mo = m + 3 - 12*int(m/10)
        yr  = d - 4800 + int(m/10)
        datestr_k=str(yr).zfill(4)+str(mo).zfill(2)+str(dy).zfill(2)
        datestr[k]=datestr_k
    if nd == 1:
        datestr=datestr[0]
    return(datestr)

## n_col = 7, n_row = 7 (for mixture and Xrec) {}
def plot_gallery(title, images, filename, v_lim, fig_col=fig_col, fig_row=fig_row, cmap=pl.cm.gist_rainbow, titles=None, xtext=None, color=None):
        #pl.figure(figsize=(2. * fig_col, 2.26 * fig_row))
        pl.figure(figsize=(1.75*fig_col, 1.75*fig_row*n_col/n_row),frameon=False)
        pl.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        pl.suptitle(title, size=12, x=0.7,y=0.05)
        if color is None:
                color=np.full(n_ifs,'red',dtype='U21');
        for i, comp in enumerate(images):
                pl.subplot(fig_row, fig_col, i + 1)
                #vmax = max(comp.max(), -comp.min())
                pl.imshow(comp.reshape(n_row, n_col).T, cmap=cmap, interpolation='nearest', vmin=v_lim[0], vmax=v_lim[1])
                pl.xticks(())
                pl.yticks(())
                if titles is not None:
                        pl.text(0.1*n_row,0.125*n_col,titles[i],fontsize=7,color=color[i])
                if xtext is not None:
                        pl.text(0.8*n_row,0.6*n_row,xtext[i],fontsize=7,color=color[i])
        #pl.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
        pl.subplots_adjust(wspace=0,hspace=0)
        pl.savefig(filename)
        
def plot_xy_gallery(title, xdate,ydata, filename, v_lim, fig_col=fig_col, fig_row=fig_row, psym='.'):
        pl.figure(figsize=(2. * fig_col, 2.26 * fig_row))
        pl.suptitle(title, size=16)
        for i, comp in enumerate(ydata):
                jds=np.mod(xdate*365.25,365.25)
                jd_mo1=np.mod(date2jd(['20200601','20200701','20200801','20200901']),365.25)
                col_v=['c','b','r','m','y']
                pl.subplot(fig_row, fig_col, i + 1)
                j0=np.where((jds<jd_mo1[0]))[0]
                j1=np.where((jds>=jd_mo1[0]) & (jds<jd_mo1[1]))[0]
                j2=np.where((jds>=jd_mo1[1]) & (jds<jd_mo1[2]))[0]
                j3=np.where((jds>=jd_mo1[2]) & (jds<jd_mo1[3]))[0]
                j4=np.where((jds>=jd_mo1[3]))[0]
                pl.plot(xdate[j0],comp[j0], psym,color=col_v[0])     #prior 1 June (cyan)
                pl.plot(xdate[j1],comp[j1], psym,color=col_v[1])     #June (blue)
                pl.plot(xdate[j2],comp[j2], psym,color=col_v[2])     #July (red)
                pl.plot(xdate[j3],comp[j3], psym,color=col_v[3])     #August (magenta)
                pl.plot(xdate[j4],comp[j4], psym,color=col_v[4])     #after 31 August (yellow)
                #pl.plot(xdate,comp, 'k:')
                #import ipdb; ipdb.set_trace()
                pl.plot(xdate,xdate*0,'k--'); 
                pl.ylim(v_lim)
                pl.xlim([np.floor(np.min(xdate)),np.ceil(np.max(xdate))])
                pl.xticks(np.arange(np.floor(np.min(xdate)),np.ceil(np.max(xdate)),step=5),rotation=90)
                pl.yticks(np.arange(np.ceil(v_lim[0]/1000)*1000, np.ceil(v_lim[1]/1000)*1000,step=1000))
        #pl.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
        pl.savefig(filename)

def make_strip(pix_start,pix_end,strip_wid=10):
        strip_start=np.array(pix_start); strip_end=np.array(pix_end); 
        strip_len=np.round(np.sqrt((strip_end[0]-strip_start[0])**2+(strip_end[1]-strip_start[1])**2)).astype(int)
        lvec=(strip_end-strip_start)/strip_len; tvec=np.array([lvec[1],-lvec[0]])
        lprof=np.linspace(0,strip_len-1,strip_len)
        tprof=np.linspace(-strip_wid/2,strip_wid/2,strip_wid)
        xstrip=strip_start[0]+lvec[0]*np.outer(lprof,np.ones(strip_wid))+tvec[0]*np.outer(np.ones(strip_len),tprof)
        ystrip=strip_start[1]+lvec[1]*np.outer(lprof,np.ones(strip_wid))+tvec[1]*np.outer(np.ones(strip_len),tprof)
        return xstrip, ystrip, lprof

def read_ifs():
        data_list = glob.glob(udir+'*.diff.unw')
        for k, file in enumerate(data_list):
                data_list[k]=os.path.basename(data_list[k])
        if len(data_list) != n_ifs:
                print('warning: file list in directory: '+udir+' has length '+str(len(data_list))+ ' vs '+str(n_ifs)+' expected! - list will be truncated (or errors out)')
        data_list=data_list[:n_ifs]; 
        n_pixels = int(n_col * n_row)
        mixture = np.empty((n_pixels,n_ifs),dtype)
        dates=np.zeros(n_ifs,dtype='U21')
        jdmid=np.zeros(n_ifs)
        period=np.zeros(n_ifs)
        multiple=np.zeros(n_ifs)
        
        #testing: calculate phase gradient in coherent region of active lobe 
        actilobe_grad=np.zeros(n_ifs,dtype='U26')
        xstrip, ystrip, lprof = make_strip([380,480],[500,420],strip_wid=15)

        if shuffle:
                print("Shuffle input data")
                np.random.shuffle(data_list)
        
        for col, file in enumerate(data_list):
                print('reading file + matching tsx-conversion factor: ' + str(col+1)+' of '+str(len(data_list)))   
                data = readBin(udir+file, (rg_samples,az_lines), dtype)

                datepair=extract_dates(file)
                jdmid[col]=(date2jd(datepair[1])+date2jd(datepair[0]))/2
                period[col]=(date2jd(datepair[1])-date2jd(datepair[0]))
                dates[col]=datepair[0]+'_'+datepair[1]

                if not center:
                        ypatch=1500; xpatch=8000; spatch2=500; 
                        jval_patch= np.where(data != 0)
                        data[jval_patch] -= np.mean(data[xpatch-spatch2:xpatch+spatch2,ypatch-spatch2:ypatch+spatch2])     

                jsens=np.where(np.remainder(period[col],sensor_periods)==0)[0][0]
                multiple[col]=period[col]/sensor_periods[jsens]
                print(sensors[jsens]+' - period: '+str(period[col])+' multiple: '+str(multiple[col]))

                convfac_file='convfac_'+file.split('_')[0]+'_tsx_lf_smooth'
                convfac = readBin(fdir+convfac_file, (rg_samples,az_lines), dtype)
                convfac *= multiple[col]
                data /= convfac
                
                data = rescale(data, 1/scale_fact,multichannel=False)
                data=data[row_start:row_start+n_row,col_start:col_start+n_col]

                mixture[:,col] = data.flatten(flatten_order)

                #testing: calculate phase gradient in coherent region of active lobe 
                #pl.imshow(data.T,cmap='jet',vmin=-10,vmax=10); pl.plot(xstrip,ystrip,'r,'); pl.show()
                ph=data[np.round(ystrip).astype(int),np.round(xstrip).astype(int)]
                jval_ph=np.where(np.sum(ph !=0, axis=1)==ph.shape[1])
                ph_av=np.mean(ph,axis=1); ph_av=ph_av[jval_ph]; lprof_av=lprof[jval_ph]
                #pl.plot(lprof,ph,'r',lprof_av,ph_av,'k'); pl.show()
                actilobe_grad[col]= dates[col]+','+str(-lsq_fit(ph_av,lprof_av,1)[0][0]*lprof[-1])
                #import ipdb; ipdb.set_trace()
        #testing: calculate phase gradient in coherent region of active lobe        
        np.savetxt('actilobe_grad.txt',actilobe_grad,fmt='%s')

        with open("unw_list.txt", "w") as fp:   #Pickling
                for inf in data_list:
                        fp.write(inf + ',')
        #np.savetxt('inf_order.txt', data_list, delimiter = ',')                
        
        if center:
                print("centered images - should only take mean of valid images (fix)")
                ### global centering
                mixT = mixture.T
                mixT_nan = np.where(mixT == 0.0, np.nan, mixT)   #nanmean warning can be ignored; is form avaraging only nan's through teh image stack in the masked out areas
                mixT_nan_mean = np.nanmean(mixT_nan, 0)
                mixture_centred = mixT - mixT_nan_mean
                
                ### local centering
                mix_nan_mean_ax1 = np.nanmean(mixture_centred, 1).reshape(n_ifs, -1)
                mixture_centred_nan = (mixture_centred - mix_nan_mean_ax1)
                mixture_centred = np.nan_to_num(mixture_centred_nan)

                mixture = mixture_centred.T
        else:
                print('uncentered images zeroed around ['+str(xpatch)+','+str(ypatch)+'] (mean of '+str(spatch2*2)+ ' pixel patch)')


        return mixture, dates, jdmid, period

def fica(mixture):
        ica = FastICA(n_components=n_comp, whiten=whiten, max_iter=iters, tol=tolerance)
        start_time = time.time()
        sources = ica.fit_transform(mixture)
        fastica_time = time.time() - start_time
        mixing = ica.mixing_  # Get estimated mixing matrix
        mean = ica.mean_
        Xrec = np.dot(sources, mixing.T) + mean
        #inverse = ica.inverse_transform(mixture)
        # assert np.allclose(X, np.dot(sources, mixing.T) + mean)
        
        # print(sources.shape)
        print(np.max(sources))
        print(np.min(sources))

        return sources, mixing, mean, Xrec, fastica_time

if __name__ == "__main__":
        
        # get mixture from interferograms
        mixture, dates, jdmid, period = read_ifs()  #units are radians of tsx leapfrog phase (all sensors are converted to tsx LOS geometry and an 11 day period)
        xxx
        idx_nz = np.where(np.abs(mixture[:,0].ravel()) != 0)[0]
        mixture_nz = mixture[idx_nz, :]



        #identify indices of "valid" ifgs (= not in omit list) used for ICA
        jval=np.where(np.in1d(dates,ica_omit_list) == False)[0]
        if jval.shape[0] != n_ifs:
                print('omitting '+str(n_ifs-jval.shape[0]) +' interferograms in data passed to FASTICA')

        # run fastica
        sources_nz, mixing_jval, mean_jval, Xrec_nz_jval, f_time = fica(mixture_nz[:,jval])
        print(f_time)
        #import ipdb; ipdb.set_trace()
        signflip=np.sign(np.max(sources_nz,axis=0)-np.abs(np.min(sources_nz,axis=0)))
        sources_nz*=signflip
        mixing=np.matmul(mixture_nz.T,sources_nz)       #!!verified that mixing is really just a simple vector projection onto the modeled sources, hence can find mixing aalso for the omitted interferograms (not used by the ICA modeling)
        mean=np.mean(mixture_nz,axis=0)
        Xrec_nz = np.dot(sources_nz, mixing.T) + mean

        ### sources_nz reconstruct
        sources = np.zeros([n_row * n_col, n_comp])
        sources[idx_nz, :] = sources_nz

        Xrec = np.zeros([n_row * n_col, n_ifs])
        Xrec[idx_nz, :] = Xrec_nz

        #write ICA results for arcinfo
        np.save("sources.npy",sources)
        np.save("mixing.npy", mixing)
        np.save("mean.npy", mean)
        np.save("recon.npy", Xrec)
        np.savetxt('mixing.txt', mixing, delimiter = ',')
        np.savetxt('dates.txt',dates,fmt='%s')
        
        #testing: calculate phase gradient in coherent region of active lobe 
        actilobe_grad_src=np.zeros(n_comp,dtype='U26')
        xstrip, ystrip, lprof = make_strip([380,480],[500,420],strip_wid=15)       
        for i in range(n_comp):
                name = 'sources_' + str(i)
                reshape_source = np.reshape(sources[:, i], (n_row, n_col))
                resc_reshape = resize(reshape_source, (rg_samples, az_lines))
                writeBin(name, resc_reshape)
                
                ph=reshape_source[np.round(ystrip).astype(int),np.round(xstrip).astype(int)]
                jval_ph=np.where(np.sum(ph !=0, axis=1)==ph.shape[1])
                ph_av=np.mean(ph,axis=1); ph_av=ph_av[jval_ph]; lprof_av=lprof[jval_ph]
                actilobe_grad_src[i]= name+','+str(-lsq_fit(ph_av,lprof_av,1)[0][0]*lprof[-1])
        np.savetxt('actilobe_grad_src.txt',actilobe_grad_src,fmt='%s')
        '''
        ## save reconstructed
        recon = np.load('recon.npy')
        for i in range(n_ifs):
                name = 'recon_' + str(i)
                reshape_recon = np.reshape(recon[:, i], (n_row, n_col))
                resc_recon = resize(reshape_recon, (rg_samples, az_lines))
                writeBin(name, resc_recon)
        '''        

        ### save mixtures.txt, save unw order
        idx=np.argsort(jdmid); 
        color=np.full(n_ifs,'red',dtype='U21'); color[jval]=np.full(jval.shape,'black',dtype='U21')
        plot_gallery("Whitened: Centered interferograms", mixture[:,idx].T, "original_ifg.png", v_lim=if_lim, titles=dates[idx],xtext=period[idx].astype(int).astype('U21'),color=color[idx])
        plot_gallery('Whitened: Centred Independent components - FastICA', sources.T, "sources.png", v_lim=s_lim, fig_col=fig_col_comp, fig_row=fig_row_comp);
        plot_gallery("Whitened: Centered interferograms - reconstructed from components", Xrec[:,idx].T, "reconstr_ifg.png", v_lim=if_lim, titles=dates[idx],xtext=period[idx].astype(int).astype('U21'),color=color[idx])
        yr=jdmid[idx]/365.25; yr=yr+1995-np.floor(np.min(yr)); source_strength=mixing[idx,:].T
        vmin=1.1*np.min(np.min(source_strength),0); vmax=1.1*np.max(source_strength); 
        plot_xy_gallery("source mixing strength", yr, source_strength, "mixing.png", v_lim=[vmin,vmax],fig_col=fig_col_comp, fig_row=fig_row_comp);
        pl.show()
