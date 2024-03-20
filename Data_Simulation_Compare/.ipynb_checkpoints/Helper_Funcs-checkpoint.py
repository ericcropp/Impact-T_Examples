# Define class for Gaussians
class GaussianParams:
    def __init__(self,*args):
        try:
            if type(args[0])==dict:
                self.Center=args[0]['Center']
                self.Amplitude=args[0]['Amplitude']
                self.Sigma=args[0]['Sigma']
                self.Baseline=args[0]['Baseline']
            else:
                self.Center=args[0][0]
                self.Amplitude=args[0][1]
                self.Sigma=args[0][2]
                self.Baseline=args[0][3]
        except:
            raise ValueError("Improperly Formatted Input")
    def evaluate_gaussian(self,x):
        try:
            return self.Amplitude * np.exp(-(x - self.Center) ** 2 / (2 * self.Sigma ** 2))+self.Baseline
        except:
            try:
                int(priors.Center)
            except:
                raise ValueError("Gaussian not defined by distribution of parameters")
    def print_params(self):
        return [self.Center,self.Amplitude,self.Sigma,self.Baseline]
        


# Filter dataframe on existence of data in column
def filter_df(df,col):
    idx=pd.isna(df[col].str.contains('NaN'))
    df_new=df[idx]
    return df_new, idx

# Make an image stack from a dataframe vector
def im_stack_from_df(df,col,ncolcol,nrowcol,fliplr=0,rot=0):
    ncols=df[ncolcol]
    nrows=df[nrowcol]
    if np.abs(np.sum(np.diff(ncols)))+np.abs(np.sum(np.diff(nrows)))==0:
        ncol=int(ncols.mean())
        nrow=int(nrows.mean())
        imgs=df[col]
        if rot:
            imgArray=np.zeros([len(imgs),nrow,ncol])
        else:
            imgArray=np.zeros([len(imgs),ncol,nrow])
        j=0
        
        for i,row in df.iterrows():
            img=imgs[i]
            img=np.reshape(img,(ncol,nrow))
            if rot:
                img=scipy.ndimage.rotate(img,90)
            if fliplr:
                img=np.fliplr(img)
            imgArray[j,:,:]=img
            j=j+1
    return imgArray

    
# Define Gaussian
def eval_gauss_baseline(p,x):  
    zx = p[2]*np.exp(-0.5*(x-p[0])**2./(p[1]**2)) +p[3]
    
    return zx

# For fitting a Gaussian
def penalty_func(p,v,x):
    zx=eval_gauss_baseline(p,x)-v
    z = np.sum(zx**2)
    return z

#Define initial guesses from image
def init_guess(xproj,lengthscale=25):
    base=np.mean(xproj[0:10])
    cx=np.argmax(xproj)
    
    tempx=np.max(xproj)
    sx=len(xproj)/lengthscale

    return GaussianParams([cx,tempx-base,sx,base])


# Fit Gaussian
def fit_gauss(xproj):
    #returns parameter of 1D gaussian fit with baseline
    #center, sigma, amplitude, baseline
    guess=init_guess(xproj)
    
    jx=np.arange(0,len(xproj),1)
    p=np.array([guess.Center,guess.Sigma,guess.Amplitude,guess.Baseline])
    res = scipy.optimize.minimize(penalty_func, p, args=(xproj,jx), method='nelder-mead', options={'xatol': 1e-5, 'disp': False})
    out=GaussianParams([res.x[0],res.x[2],res.x[1],res.x[3]])
    return out
# this function rotates the image 45 degrees CCW
def imrotate45(img,bg):
    img1=Image.fromarray(img)
    img2=img1.rotate(45,fillcolor=int(bg))
    img3=np.array(img2)
    return img3

# Runs through 4 image fits for a single image
def Gaussian_Fit_4_Dim(img1):
    
    xproj=np.sum(img1,0)
    yproj=np.sum(img1,1)
    xfit=fit_gauss(xproj)
    yfit=fit_gauss(yproj)

    bg=(xfit.Baseline/np.shape(img1)[0]+yfit.Baseline/np.shape(img1)[1])/2
    img3=imrotate45(img1,bg)
    xproj45=np.sum(img3,0)
    yproj45=np.sum(img3,1)
    
    xfit45=fit_gauss(xproj45)
    yfit45=fit_gauss(yproj45)

    return xfit, yfit,xfit45,yfit45

# Calculates stats for multiple images of a specific setting
def calc_img_stats(avg_setting_list,setting_list,gauss_list):
    if len(np.shape(avg_setting_list))>1 or len(np.shape(setting_list))>1:
        raise ValueError("too many dimensions to average")
        
    else:
        means=np.zeros(np.shape(avg_setting_list))
        stds=np.zeros(np.shape(avg_setting_list))
        avg_setting_list=np.array(avg_setting_list)
        setting_list=np.array(setting_list)
        gauss_list=np.array(gauss_list)
        
        for i in range(len(avg_setting_list)):
            idx=np.array(setting_list)==np.array(avg_setting_list[i])
            new_list = [x.print_params()[2] for x in gauss_list[idx]]
            
            means[i]=np.mean(new_list)
            
            stds[i]=np.std(new_list)

    return means, stds

# Fits for every image in a stack
def imgArray_Fits(imgArray):
    fits=[]
    for i in range(np.shape(imgArray)[0]):
        xfit, yfit,xfit45,yfit45=Gaussian_Fit_4_Dim(imgArray[i,:,:])
        fits.append((xfit, yfit,xfit45,yfit45))
    fits=tuple(zip(*fits))

    return fits

# Make an image from a simulation
def make_img_particles_hist(x1,x2,range):
    h,xedges,yedges=np.histogram2d(x1,x2,bins=int(np.sqrt(len(x1)/10)),range=range);
    h=np.flipud(h) # histogram origin at bottom, image at top
            
    return h


# For plotting: Centers an image
def Center_Image(img,pixcal,xCent,yCent):
    pxsz=np.array(np.shape(img))

    extent1=pxsz*pixcal
    xCentoff=xCent*pixcal-extent1[0]/2
    yCentoff=yCent*pixcal-extent1[1]/2
    
    extent=[(extent1[0]/-2)-xCentoff,(extent1[0]/2)-xCentoff, (extent1[1]/-2)+yCentoff,(extent1[1]/2)+yCentoff]
    return extent

#For plotting: gives smoothed max pixel
def Max_Pixel(img):
    img2 = gaussian_filter(img,sigma=2)
    m=np.sort(np.reshape(img2,[1,250000]))[-1][-1]
    return m
