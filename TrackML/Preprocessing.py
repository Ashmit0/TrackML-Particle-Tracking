# import libraries : 
import os 
import pickle
import numpy as np 
import pandas as pd 

import torch 
from torch import Tensor 

#### Preprocessing Detector Data #### 


def process_detector_data(detector_path:str)->dict: 
    '''
    param detector_path: path like string pointing to
        the path of the detector csv file. 
    returns detector: a dictionary object containgin the rotational matrics 
        pitch and thickness for all each volume and layer id. 
    '''
    
    # read the dectector data frame 
    detector_df = pd.read_csv( detector_path )
    
    # first get max vallues of the volume_id , layer_id and module_id 
    vid_max , lid_max , mid_max = (
        detector_df.volume_id.max() + 1 , 
        detector_df.layer_id.max()  + 1 , 
        detector_df.module_id.max() + 1 
    )
    
    # initialize the result dictionary 
    detector = {} 
    
    # initialize the arrays : 
    # for thickness, pitch_u and pitch_v of the module
    detector['thickness'] ,detector['pitch_u'] , detector['pitch_v'] = (np.zeros(shape=(vid_max,lid_max,mid_max)) for _ in range(3))
    # rotation matrices from local to grobal coordinates : 
    detector['rotation'] = np.zeros(shape=(vid_max,lid_max,mid_max,3,3))
    # also store the total number of unnique modules : 
    detector['module_count'] = detector_df.shape[0]
    
    # memoization : 
    for _ , r in detector_df.iterrows() : 
        # print( r[['volume_id','layer_id','module_id']].to_numpy(dtype = int) )  
        v , l , m = r[['volume_id','layer_id','module_id']].to_numpy(dtype = int)
        
        detector['thickness'][v,l,m] = 2*r.module_t 
        detector['pitch_u'][v,l,m] = r.pitch_u  
        detector['pitch_v'][v,l,m] = r.pitch_v 
        detector['rotation'][v,l,m] = np.array([
            [r.rot_xu,r.rot_xv,r.rot_xw], 
            [r.rot_yu,r.rot_yv,r.rot_yw], 
            [r.rot_zu,r.rot_zv,r.rot_zw]
        ])
        
    del detector_df 
        
    return detector


def load_detector_data(detector_path:str)->dict: 
    detector_pickle_path = os.path.splitext(detector_path)[0] + '.pickle'
    try: 
        with open(detector_pickle_path,"rb") as f:
            detector = pickle.load(f)
    except: 
        print('Preprocessing Detector Data ... ' )
        print('Note this is only a one time process')
        detector = process_detector_data(detector_path) 
        with open( detector_pickle_path , 'wb' ) as f : 
            pickle.dump(detector , f ) 
        print(f'Detector Preprocessing Complete.\nThe preprocessed dictionaty is saved at {detector_pickle_path}.')
    return detector 

#### General Functions #### 

def Cartesian_to_Spherical(x,y,z): 
    r = np.sqrt( x*x + y*y + z*z ) 
    rxy = np.sqrt( x*x + y*y )
    theta = np.arccos(z/r)
    sphi  = y/rxy 
    cphi  = x/rxy 
    del rxy 
    return r,theta,sphi,cphi 

def Cartesian_to_Cylindrical(x,y,z): 
    rxy = np.sqrt( x*x + y*y ) 
    # sin phi 
    sinphi = y/rxy 
    cosphi = x/rxy 
    return rxy , sinphi , cosphi , z  


#### Preprocess an Event Data #### 


def get_local_differential_volume(processed_event:pd.DataFrame,hits:pd.DataFrame,cells:pd.DataFrame,detector:dict)->None: 
    '''
    lu , lv , lw: local diffrential direction of the hit 
        regestired on the detector. 
    '''
    # get the volume , layer and module ids for each event : 
    vid , lid , mid = hits.loc[ : , ['volume_id' , 'layer_id' , 'module_id']].values.T 
    
    # extent of channel 0 and 1 pixels (u-axis and v-axis respectively)
    foo =cells.groupby('hit_id').ch0.agg(['max', 'min']).values 
    processed_event['lu'] = (foo[:,0]-foo[:,1]+1)*detector['pitch_u'][vid,lid,mid]
    foo=cells.groupby('hit_id').ch1.agg(['max', 'min']).values 
    processed_event['lv'] = (foo[:,0]-foo[:,1]+1)*detector['pitch_v'][vid,lid,mid] 
    
    # thickness of the detector ( perpendicular to the plane of the module ): 
    processed_event['lw'] = detector['thickness'][vid,lid,mid]
    
    del foo , vid , mid , lid
    

    
def get_orentations(processed_event:pd.DataFrame,hits:pd.DataFrame,cells:pd.DataFrame,detector:dict)->None: 
    '''
    gsphi , gcphi: sin and cos of the global diffential 
        spatial hit direction in spherical coordinate system. 
    gtheta, ltheta: global and local angle of the diffential 
        spatial hit direction in spherical coordinates. 
    lsphi , lcphi:  sin and cos of the global diffential 
        spatial hit direction in spherical coordinate system.  
    '''
    
    # get this volume , layer and module id:  
    vid , lid , mid  = hits.loc[:, ['volume_id' , 'layer_id' , 'module_id' ] ].values.T 
    
    # get the local spatial differential vector
    lu , lv , lw = processed_event.loc[ : , ['lu' , 'lv' , 'lw']].values.T 
    
    # for local spherical angles : 
    (
        _ , 
        processed_event['ltheta'] , 
        processed_event['lsphi'] , 
        processed_event['lcphi'] 
        
    ) = Cartesian_to_Spherical(lu,lv,lw)
    
    # get the rotations: 
    rotations = detector['rotation'][vid,lid,mid]
    
    # format the dircetions and convert them from 
    # local to  global frame of refrence: 
    lu = np.expand_dims( lu , axis = 1 )
    lv = np.expand_dims( lv , axis = 1 )
    lw = np.expand_dims( lw , axis = 1 )
    directions = np.concatenate((lu,lv,lw),axis=1)
    directions = np.expand_dims(directions,axis=2)
    
    # get the rotated vectors : 
    global_directions = np.matmul(rotations,directions).squeeze(2)
    del lu , lv , lw , directions , vid , lid , mid , rotations  
    
    # convert and store as angles in spherical coordinates : 
    (
        _ , 
        processed_event['gtheta']  , 
        processed_event['gsphi'] , 
        processed_event['gcphi']  
    ) = Cartesian_to_Spherical(*global_directions.T)
    del global_directions 
    
    
def process_event_data(train_path:str,eventid:str,detector:dict,return_dataframe:bool=False): 
    '''
    train_path : path to the folder containing the csv files of the 
        evenid to process. 
    eventid : string containg the unique eventid. 
    return_dataframe : if true then return a Pandas Dataframe object, otherwise 
        return as pytorch tensor of the featuers excluding the hit id. 
    '''
    
    hits = pd.read_csv(train_path+eventid+'-hits.csv')
    cells = pd.read_csv(train_path+eventid+'-cells.csv')

    # initialize the processed event dataframe : 
    processed_event = hits.loc[: , ['hit_id' ]]
    # global spatial position of the hits :
    (
        processed_event['r'], 
        processed_event['sphi'],
        processed_event['cphi'],
        processed_event['z']
    ) = Cartesian_to_Cylindrical(*hits.loc[ : , ['x' , 'y' , 'z']].to_numpy().T)
    # number of cells activated 
    processed_event['ncells'] = cells.groupby('hit_id').hit_id.count().values.copy() 
    # total signal value deposited 
    processed_event['tvalue'] = cells.groupby('hit_id').value.sum().values.copy()
    
    get_local_differential_volume(processed_event,hits,cells,detector)
    
    get_orentations(processed_event,hits,cells,detector)
    
    if return_dataframe : 
        return processed_event
    else  : 
        return Tensor( processed_event.iloc[: , 1: ].values )
    

def process_particle_labels(train_path:str,eventid:str,min_nhits:int=3)->torch.tensor: 
    '''
    return the particlesids for each event from the truch.csv file 
    particles with hits less then min_hits are treated with fake hits. 
    '''
    # read the truth.csv file 
    truth = pd.read_csv(train_path+eventid+'-truth.csv')

    # add a new column nhits : the number of hits a particle leaves on the detector 
    truth['nhits'] = truth.particle_id.map(truth.particle_id.value_counts())

    # filter the truth file such that if the 
    # number of the hits is less then the min then the
    # particle id is updated to 0. i.e. a fake hit 
    truth.loc[truth.nhits < min_nhits , 'particle_id' ] = 0 
    
    return torch.tensor(truth.particle_id.values)


def get_track_index_pairs(particle_ids:torch.tensor)->torch.tensor: 
    
    unique_id = torch.unique(
        particle_ids, return_inverse=False, sorted=False
    )
    
    row , col  =  [] , []
    for uid in unique_id:
        if( uid == 0 ): 
            continue
        mask = ( uid == particle_ids )
        indices = torch.nonzero(mask, as_tuple=True)[0]
        n = indices.size(0)
        
        if n > 1:
            # Generate all unique index pairs using broadcasting : 
            row_idx, col_idx = torch.triu_indices(n, n, offset=1) 
            row.extend( indices[row_idx].tolist() )
            col.extend( indices[col_idx].tolist() )
    
    # create tensors for row and column : 
    row , col = torch.tensor(row).unsqueeze(dim=0) , torch.tensor(col).unsqueeze(dim=0) 
    
    # concatinate to make a edge_index type tensor object : 
    return torch.concatenate((row,col),dim=0) 
    