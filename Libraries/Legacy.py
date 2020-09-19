#def RHTPlanes(points)
load=False
if load==False:
    restart=False
    if restart==True:
        workpoints=pointsDF
        found=[]

    maxtheta=np.pi
    mintheta=-np.pi
    maxp=1
    minp=-1

    sizeofac=6
    acumulator=np.zeros([sizeofac,sizeofac,sizeofac])
    threshhold=40
    acumulatorchoices=np.zeros([sizeofac,sizeofac,sizeofac,(threshhold+1)*3])
    xstep=(maxtheta-mintheta)/sizeofac
    ystep=xstep
    zstep=(maxp-minp)/sizeofac
    mindist=0.08
    distance=0.15
    maxiterations=400000
    iteration=0
    while len(workpoints)>2:
        iteration+=1
        choices=np.random.choice(len(workpoints),3)
        points=workpoints[choices]
    
        if iteration%50000==0:
            print(iteration)
    
        if iteration>maxiterations:
            break
        
        if ((np.linalg.norm(points[0]-points[1])<distance)
        & (np.linalg.norm(points[2]-points[1])<distance) 
        & (np.linalg.norm(points[2]-points[0])<distance)
        & (np.linalg.norm(points[0]-points[1])>mindist)
        & (np.linalg.norm(points[2]-points[1])>mindist) 
        & (np.linalg.norm(points[2]-points[0])>mindist)):
            P=np.dot(np.cross(points[2]-points[1],points[0]-points[1]),points[0])
            x0=[0,0]
            arges = scipy.optimize.least_squares(Corners.hugh3p_inter, x0, args=(points[0],points[1],points[2]), \
                                                 ftol =1e-6,xtol =1e-6, \
                                         gtol =1e-6,max_nfev =10,bounds=[[-np.pi,-np.pi],[np.pi,np.pi]])
            if arges.success==True:
                theta=int((arges.x[0]+maxtheta)/xstep)
                phi=int((arges.x[1]+maxtheta)/ystep)
                #p=int((hughtransform_1p(points[0],arges.x[0],arges.x[1])+maxp)/(zstep))
                p=int((P+maxp)/(zstep))
            
                acumulatorchoices[theta,phi,p,int(acumulator[theta,phi,p]*3)]=choices[0]
                acumulatorchoices[theta,phi,p,int(acumulator[theta,phi,p]*3+1)]=choices[1]
                acumulatorchoices[theta,phi,p,int(acumulator[theta,phi,p]*3+2)]=choices[2]
                
                acumulator[theta,phi,p]+=1
            
                if acumulator[theta,phi,p]>threshhold:
                    print('planefound')
                    choicepoints=np.int32(acumulatorchoices[theta,phi,p,:])
                    foundpoints=workpoints[np.int32(acumulatorchoices[theta,phi,p,:])]
                    found.append([theta,phi,p,foundpoints])
                    #planepoint
                    workpoints=np.delete(workpoints,choicepoints,0)
                    acumulator=np.zeros([sizeofac,sizeofac,sizeofac])
                    acumulatorchoices=np.zeros([sizeofac,sizeofac,sizeofac,(threshhold+1)*3])
    print('points left',workpoints.shape)
    print('planes found',len(found))
    
    
if load==False:
    mypoints=[]
    my_list=[]
    my_points=[]
    for i in range(len(found)):
        my_list.append(found[i][0:3])
        my_points.append(found[i][3])
    
    myarr=np.array(my_list)
    uniques=np.unique(myarr,axis=0)
    print(uniques)
    for o in range(len(uniques)):
        mypoints.append([])

    for i in range(len(myarr)):
        for o in range(len(uniques)):
            if np.array_equal(myarr[i],uniques[o]):
                if len(mypoints[o])==0:
                    mypoints[o].append(my_points[i])
                else:
                    mypoints[o][0]=np.concatenate([mypoints[o][0],my_points[i]],0)
    totalpoints=0
    for o in range(len(uniques)):
        totalpoints+=len(mypoints[o][0])
        print(len(mypoints[o][0]))
    
    planemean=totalpoints/len(uniques)
    posibleplanes=[]
    for o in range(len(uniques)):
        if len(mypoints[o][0])>planemean:
            posibleplanes.append(mypoints[o][0])
            
if load==False:
    if len(uniques)<=20:
        vis = open3d.Visualizer()
        vis.create_window()
        for i in range(len(uniques)):
            colNORM=i/len(uniques)
            color=cm.jet(colNORM)[:3]
            pointcloud1=Corners.convertcloud(mypoints[i][0],Path)
            pointcloud1.paint_uniform_color(color)
            vis.add_geometry(pointcloud1)
        vis.run()    
        vis.destroy_window()
    else:
        vis = open3d.Visualizer()
        vis.create_window()
        for i in range(len(posibleplanes)):
            colNORM=i/len(posibleplanes)
            color=cm.jet(colNORM)[:3]
            pointcloud1=Corners.convertcloud(posibleplanes[i],Path)
            pointcloud1.paint_uniform_color(color)
            vis.add_geometry(pointcloud1)
        vis.run()    
        vis.destroy_window()
        
if load==False:
    vis = open3d.Visualizer()
    vis.create_window()
    layers=[0,1]
    for i in layers:
        colNORM=i/5
        color=cm.jet(colNORM)[:3]
        pointcloud1=Corners.convertcloud(mypoints[i][0],Path)
        pointcloud1.paint_uniform_color(color)
        vis.add_geometry(pointcloud1)
    vis.run()    
    vis.destroy_window()
    

#OLD PLANE DETECTION
planenormals = []
planecentroid = []
planes = []
values = []

vectorspace,vectorcentroid,vectorspacepoints = Plane.FindPlanessparse(BoundDF)
for i in vectorspacepoints:
    nfpp = np.array([o for o in i])
    orig = len(nfpp)
    while len(nfpp)/orig>0.3:
        out = Corners.FitPlaneNormalValues(nfpp,thresh=0.4)
        nfpp = out[3]
        if len(out[2])>0:
            planecentroid.append(out[4])
            planenormals.append(out[0])
            planes.append(out[2])
            values.append(out[1])
        else:
            break
len(planes),len(vectorspacepoints)


finalplanes = []
badplanepoints = []
bufferplanepoints = planepoints
for i in range(len(planenormals)):
    planepointdist = Plane.findPlanePointswithcentroid(planenormals[i],planecentroid[i],bufferplanepoints)
    cond = np.array(planepointdist)<0.06
    goodplanepoints = bufferplanepoints[cond]
    badplanepoints = +bufferplanepoints[cond]
    if len(goodplanepoints)>100:
        bufferplanepoints = bufferplanepoints[~cond]
        finalplanes.append(goodplanepoints)
Corners.open3dpaint(finalplanes,axis=True)