import os
#将数据位置及label存入txt文件并返回文件路径
def GetImgPath(Img_path,path_filename):
    path_list=os.listdir(Img_path)
    i=0
    path_file=os.path.join(os.path.dirname(Img_path),path_filename)
    with open(path_file,'w') as f:
        for path in path_list:
            whole_path=os.path.join(Img_path,path)
            file_list=os.listdir(whole_path)
            for file in file_list:
                file_path=os.path.join(whole_path,file)
                f.write(file_path+' '+str(i)+'\n')
            i=i+1
    return path_file

def GetImgInfo(path_file):
    imgs=[]
    with open(path_file) as f:
        lines=f.readlines()
        for line in lines:
            Info=line.rstrip().split()
            imgs.append({'path':Info[0],'label':int(Info[1])})
    return imgs

