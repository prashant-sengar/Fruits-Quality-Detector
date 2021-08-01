from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from fastai import *
from fastai.vision import *
# Create your views here.
#Path to the dataset
path =r'C:\Users\Prashant\Desktop\My datasets\dataset\dataset\train'
np.random.seed(101)
data=ImageDataBunch.from_folder(path, train=".", valid_pct=0.3, ds_tfms=get_transforms(), size=224, num_workers=4,bs=8).normalize(imagenet_stats)
mymodel=cnn_learner(data,models.resnet50,metrics=accuracy)
##Load your model
mymodel.load(r'C:\Users\Prashant\Desktop\My datasets\dataset\dataset\model\model-1)
def index(request):
    context={'a':1}
    return render(request,'index.html',context)

def classify(request):
    if request.method == 'POST':
        obj=request.FILES["filepath"]
        fb=FileSystemStorage()
        filepathname=fb.save(obj.name,obj)
        #print(filepathname)
        img=open_image(filepathname)
        pred_class,ab,cd=mymodel.predict(img)
        result="The predicted class is "
        context={'filepathname':filepathname,'prediction':result,'class':pred_class}
    else :
        context={'a':1}
    return render(request,'index.html',context)

