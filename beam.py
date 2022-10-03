#!/usr/bin/env python
# coding: utf-8

def about():
    print(" Current version:", __version__)
    print(" Latest update:", __latest_update__)

__latest_update__='20/9/2022'

__version__='2.10.3.0'

__update_history__="""
Version 2.6.x
- adding trend based function
- agent now can set their process to be trend based
- builtin metrics (stockflow, avg_sigma)
- mkt.add_part_sentiment_fromshared
- time-series utils (mav, plot smoothing)
- adding MAV based function
- agent now can set their process to be MAV based

Version 2.7.x
- introducing simulation mode "exogenous"
- initial sigma and totwealth of agents can now be randomly initialized
- mkt.plot_state_dist for plotting agent state distributions
- every agent process(sentiment, prediction,...) is generalized to be sequence of subprocesses
- additive vs assignative process option (operation parameter in builtin process)
- timers + program runs 2 times faster :D
- add vs set in structure of subprocesses (_struct_op in mkt. set/add sentiment_by...)
- implemented Subprocess class (31/12/21)

Version 2.8.x
- impemented Lab
- lab can run variable instance (param_run)
- modularization of decision process
- every process is now an istance of subprocess
- agent does not take any default_{process}_kwargs when initialized anymore
- working on Process Concept
- implemented Process_class (3,1,22)
- implemented OPERATION_MODE= 'class' or 'fn' ('class' works with/provides metadata while 'fn' 20% faster)
- implemented Process.MODE= 'implicit'

Version 2.9.x
- implemented fit functions for Lab objects
- plotting and manipulation of fit and metrics (inizio gennaio 2022, pausa)
- introduced default_decision_kw['subaction_par'] (inizio marzo 2022, esame)
- sistemazioni per report di esame + funzione coefficienti dei fit ha modalità distr e scatter

Version 2.10.x
0 implemented array listed sentiment values (parallel processing of sentiment) (22/08/22)
1 implemented onesided gaussian price sampling
1 implemented operation 'map' in predefined sentiment process
1 implemented (re) subaction market and kind=noise agents
2 generalized process management
2 agents now all have state, process, parameter dictionaries
2 added activate_beta for behaviour of onesided gaussian 
3 added decision_stop_days
"""

###TO DO
# - wrap every process
# - use uniform for noisy agents (?)
# - subaction (subchoice) as an operation on its own

import numpy as np
from numpy.random import beta as Beta 
import matplotlib. pyplot as plt

import random as random

from tqdm.notebook import tqdm

import os    
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from datetime import date
from datetime import timedelta

# import mplfinance  as mpl
# import networkx as nx
import pandas as pd

from scipy.stats import beta

from time import time
from copy import copy, deepcopy

# TIMERS={} not worth it atm
TEMP_TIMERS={}

def reset_timers():
#     global TIMERS
    global TEMP_TIMERS
#     TIMERS=deepcopy(TEMP_TIMERS)
    for timer_name in TEMP_TIMERS:
        TEMP_TIMERS[timer_name]=0

def dir_timers(timer_name='all'):
    if timer_name=='all':
        print("  Latest run timer info:")
        print()
    #     for timer_name, time in TIMERS.items():
        for timer_name, time in TEMP_TIMERS.items():
            if time>0.:
                print("    {:_<55}:{:_>20.6f} sec".format(timer_name, time))
    else:
        print("    {:_<55}:{:_>20.6f} sec".format(timer_name, TEMP_TIMERS[timer_name]) )
    
def timer_decorator(timer_name='auto'):
    global TEMP_TIMERS
    if timer_name=='auto':
        name='timer_'+str(len(TEMP_TIMERS))
    if timer_name not in TEMP_TIMERS:
        TEMP_TIMERS[timer_name]=0.
#     if timer_name not in TIMERS:
#         TIMERS[timer_name]=0.
    def timer(func):
        def inner(*args, **kwargs):
            start=time()
            to_return=func(*args, **kwargs)
            end=time()
#             TIMERS[timer_name]+=end-start
            TEMP_TIMERS[timer_name]+=end-start
            return to_return
        return inner 
    return timer

timer_market_run         =timer_decorator('market_run')
timer_shared_computation =timer_decorator('--shared_computation')
timer_market_metric      =timer_decorator('--market_metric')
timer_market_step        =timer_decorator('--market_step')

timer_agent_tot_preliminar    =timer_decorator('----agent_tot_preliminar')
timer_agent_tot_thought       =timer_decorator('----agent_tot_thought')
timer_agent_tot_action        =timer_decorator('----agent_tot_action')
timer_agent_def_decision      =timer_decorator('------agent_def_decision')
timer_agent_def_sample_choice =timer_decorator('--------agent_def_sample_choice')
timer_agent_def_sample_price  =timer_decorator('--------agent_def_sample_price')
timer_agent_def_sample_wealth =timer_decorator('--------agent_def_sample_wealth')
timer_agent_def_action        =timer_decorator('------agent_def_action')

timer_book_management      =timer_decorator('--------book_order_processing')
timer_book_order_appending =timer_decorator('----------book_order_appending')
timer_book_memory          =timer_decorator('----------book_transaction_memory')

VERBOSITY_MODE=True #not properly implemented

def null_kwfn(self, **kwargs):
    return None

OPERATION_MODE='class'
# SUBPROCESS_MODE='function'


#this is basically a way to instance a Process class of subtype operation (when SUBPROCESS_MODE is class)
#or just take back the given function
def Operation(*args, **kwargs): #<-------- name needs to be changes in Operation
    obj= Process(*args, **kwargs)
    if len(obj)>1:
        raise TypeError("Operation requires one function only")
    if OPERATION_MODE.lower() =='class':
        return obj
    if OPERATION_MODE.lower() =='function':  
        return obj.get_all_functions()[0]
    
def get_operation_mode():
    print(f" current {OPERATION_MODE=}")
    return OPERATION_MODE

def set_operation_mode(mode):    #<----  class / fn 
    global OPERATION_MODE
    print(f" current {OPERATION_MODE=}")
    if mode.lower() in ['cl','class']:
        OPERATION_MODE='class'
        print(f" Set {OPERATION_MODE=}")
    elif mode.lower() in ['fn','func', 'function']:
        OPERATION_MODE='function'
        print(f" Set {OPERATION_MODE=}")
    else:
        print(f" subprocess_mode {mode} not found")
    _re_instantiate_default_fn() #need reinstantiation since def_{pr} are objects

def get_Process_def_mode():  
    return Process.get_mode()

def set_Process_def_mode(mode):  #<----  implicit / explicit 
    return Process.set_mode(mode)

# operation mode determines wheter 
# -  ('class')    to use the Process class 
# -  ('function') to use the naked functions

# Process_def_mode mode is active only when operation mode is set to class
# since determines the default initialization mode for Process instance
# every instance of Process has its own value for mode,
# so his behaviour can be overwritten
# operation protocol mode determines wheter 
# -  ('explicit') the operation are called -and need to be defined as- as (actor, **ext_kw) 
# -  ('implicit') the operation are called -and need to be defined as- as (arguments, ..., **ext_kw)
# NB default process are instanced as 'explicit'

class Process:
################################################################################ class management 
    DEBUG_PRINT=False
    
    _allowed_mode=['explicit', 'implicit'] 
    
    #implicitly execute any operation as function(*internal_input, **external_input)
    # where input is retrieved by name from agent variable
    # and output set agent parameters (with the name of the process) with function return value 
#     MODE='implicit' 

    #explicitly execute any operation as function(actor, **kw)
    MODE='explicit'
    
    def set_mode(mode):
        if mode in Process._allowed_mode:
            Process.MODE=mode
        else:
            print(f" unknown mode {mode}. Choose one in {_allowed_mode}")
    
    def get_mode():
        return Process.MODE
    
    
    _allowed_subtypes=['uninitialized', 
                        'operation', 
#                         'operation_sequence', #<----- depreacted in 1.5
#                         'comp_process',  #<---- deprecated in 1.3
                        'process']
    
    _key_sep='_S_'
    #<--- when autonaming an operation through the name of a function
    #     the program will recognise the string after the _key_sep as a sub_name (after first key_sep) 
    #     and metadata if function name contains more than 1 key_separator
    def set_key_subname_separator(key_sep):
        if not isinstance(key_sep, str):
            raise TypeError("provided key subname separator must be a string")
        if len(key_sep)>5:
            print("WARNING suggested maximum lenght of key subname separator is 5")
            return
        Process._key_sep=key_sep
    
    def get_key_subname_separator():
        return Process._key_sep
    
    KEEP_TRACK=False #could even leave it as only possible behaviour
    #<------ if True, names will be checked every new instance (slower) and instances dict updated every time
    # if False, names will never be checked (faster) and instances dict will be updated only at dir_instances call
    
    _all_obj=list()
    instances={subtype_name:{} for subtype_name in _allowed_subtypes}
    
    def __add_new_instance(new_obj):     #gets called every time a new object is initialized
        
        if Process.DEBUG_PRINT: print(f"adding {new_obj.name} to global memory of all instances") 
        Process._all_obj.append(new_obj)
        if not Process.KEEP_TRACK: return
        Process._checkset_classtatus()
        return True
    
    # THIS FUNCTION IS RESPONSABLE OF MANAGING AUTO NAMES (together with __eq__ operator)
    # procedure for checking present names and getting a valid one (eventually with suffix)
    def _checkget_name(obj, new_name):
        if not Process.KEEP_TRACK: return new_name #<-----if not keeping track of objects, the name is always ok
        if Process.DEBUG_PRINT: print(f"     checking  {new_name}")
            
        obj.name=new_name   #<----temporary name assignation for making use of == operator
        name_is_valid=True
        for other_obj in Process._all_obj: #name is not valid only if there are non equal instances with same name
            if obj.__control_id == other_obj.__control_id:
                continue
            if obj.name == other_obj.name:
                if obj.type != other_obj.type:
                    continue
                if not obj==other_obj:
#                     print(obj)
#                     print(other_obj)
                    name_is_valid=False
                    if Process.DEBUG_PRINT: print(f"        name {new_name} is not valid")
                    break
                   
        if name_is_valid: 
            return new_name

        i=1
        while not name_is_valid:  #<--- could be defined recursively but could  not (easily) return the name in that case
            i+=1  #starts from 2
            temp_name=new_name+'_'+str(i) #<------ if provided name was not valid, appends a suffix and iterate validity check
            if Process.DEBUG_PRINT: print(f"     checking  {temp_name}")
            obj.name=temp_name
            name_is_valid=True
            for other_obj in Process._all_obj: #name is not valid only if there are non equal instances with same name
                if obj.__control_id == other_obj.__control_id:
                    continue
                if obj.name == other_obj.name:
                    if obj.type != other_obj.type:
                        continue
                    if not obj==other_obj:
    #                     print(obj)
    #                     print(other_obj)
                        name_is_valid=False
                        if Process.DEBUG_PRINT: print(f"        name {temp_name} is not valid")
                        break

        new_name=new_name+'_'+str(i)

        return new_name
                         
    # gets called when a process is initialized with no user provided name
    # gets called when an auto named operation is updated with add_process(valid operation or process)
    def _setget_auto_name(obj): 
        if Process.DEBUG_PRINT: print(f"  finding an autoname ")

        if obj._name_status=='user_named': #<---- should never happend
            raise TypeError(f"some error in instancing {obj} name")
        pr_type=obj.type
#         new_name=pr_type+'_'+str(len(Process.instances[pr_type])+1)
        new_name='unnamed_'+pr_type
        name_checked=Process._checkget_name(obj, new_name)
        obj.name=name_checked
        if Process.DEBUG_PRINT: print(f"       assigned name {name_checked}")
        return name_checked
    
    # gets called whenever a process name is changed
    #  which can happend when the user calls process_obj.set_name(valid)
    #  or when an auto named operation is updated with add_process(valid operation or process)
    def _build_dict_routine():
        #------------- refresh dictionary
        Process.instances={subtype:{} for subtype in Process._allowed_subtypes}

        for obj in Process._all_obj:
            if obj.name not in Process.instances[obj.type]:   #{type:{name:[[],[],....] } }
                Process.instances[obj.type][obj.name]=[[obj]] #<--- creates new name memory 
            else:
                found_branch=False
                for branch_k, branch_obj in enumerate(Process.instances[obj.type][obj.name]):
                    if obj==branch_obj[0]:
                        Process.instances[obj.type][obj.name][branch_k].append(obj) #<--- add to existing branch
                        found_branch=True
                if not found_branch:
                    Process.instances[obj.type][obj.name].append([obj]) #<--- create new branch for esisting name
    
    def _checkset_classtatus():
        if not Process.KEEP_TRACK: return
        if Process.DEBUG_PRINT: print(f"checking all instances integrity\n   rebuilding instances dict ")          
        Process._build_dict_routine() #<--- faster process needs to be implemented


                    
            
    def dir_instances():
        if not Process.KEEP_TRACK:
            Process._build_dict_routine()  #<---- otherwise it is empty

        for inst_type, inst_name_dict in Process.instances.items():
            N_branches=0
            N_tot_obj =0
            for inst_branch_list in inst_name_dict.values():
                N_branches+=len(inst_branch_list)
                for inst_branch_sublist in inst_branch_list:
                    N_tot_obj+=len(inst_branch_sublist)
            print() 
            print("  {:_>4}_{:_<55}{:_<15}__{:_<14}".format(
                len(inst_name_dict),inst_type+'_names', 'tot_branches:_'+str(N_branches),'tot_obj:_'+str(N_tot_obj)) ) 
            for inst_name, inst_branch_list in inst_name_dict.items():

                for branch_k, inst_branch_sublist in enumerate(inst_branch_list):
                    N_obj_in_branch=len(inst_branch_sublist)
                    inst_obj=inst_branch_sublist[0]
#                     print(N_branches, N_obj_in_branch)
                    print( "  _____{:_<60}__branch_{:_<3}_of_size_{:_<5}".format(str(inst_obj), branch_k, N_obj_in_branch))
#                     print()
#                     print(f"{branch_k=}, {inst_branch_list=}")
#                 inst_obj=inst_obj[0]
# #                 print(f"        global name: {inst_name}")

                if inst_name != inst_obj.name:
                    print(f"global name {inst_name=} while {inst_obj.name=} (not coordinated)")

    
################################################################################ instance basics 
        
    def __len__(self):
        try:
            return len(self.linked_process)
        except:
            lp_type=type(self.linked_process)
            if not isinstance(self.linked_process, (tuple, list)):
                lp_type=type(self.linked_process)
                raise TypeError(f" {self.name}.linked_process is of type {lp_type}. "
                                +"Should instead be tuple or list")
                
    def __iter__(self):
        for linked_process in self.linked_process:
            yield linked_process
            
    def __repr__(self):
        repr_str=f"{self.type[:2]}. {self.name}"#({self._name_status})"
        if self.sub_name is not None:
            repr_str+=f": {self.sub_name}" #<-------- using 'sub name' for specific op name 
        if len(self.metadata):
            if len(self.metadata)<6:
                repr_str+=' ('
                dir_metadata=[str(val)[:8] for val in self.metadata.values()]
                repr_str+=",".join(dir_metadata)
                repr_str+=')'
        
#         if len(self)>1:
#             repr_str+=f" lenght {len(self)}"
        return repr_str
    
#     def __hash__(self, idx):
#         return self.linked_process[idx]
    
    def __eq__(self, other):
        #this could fail in case there are same named function with different behaviours (return True)
        #or differently named that have same behaviour (return False)
        if isinstance(self, Process):
            if not isinstance(other, Process):  
                print("qui?")
                return False
        if not self.name==other.name:
            return False
        if not self.type==other.type:
            return False
        if not len(self)==len(other):
            return False
        if self.metadata!= other.metadata:
            return False
        for l_p_s, l_p_o in zip(self, other): #could be defined recursively
            if isinstance(l_p_s, Process) and isinstance(l_p_o, Process): 
                if not l_p_s.name==l_p_o.name:    #but there is a possibility of infinite recursion
                    return False                  
                if not l_p_s.type==l_p_o.type:
                    return False
                if not len(l_p_s)==len(l_p_o):
                    return False
            elif isinstance(l_p_s, Process) or isinstance(l_p_o, Process): #if only one of these is process instance
                return False
            else:
                if not l_p_s==l_p_s:
                    return False
        return True
                        
    #only fails if a process is called INSIDE a function, 
    #which should not be done as it is against the concept of Process
    def _check_recursion(self, to_check=None):
        if to_check is None:
            if Process.DEBUG_PRINT: print(f"checking recursion of {self.name}")
        else:
            if Process.DEBUG_PRINT: print(f"    checking recursion of {self.name}: comparing with {to_check.name}")

        if to_check is None:
            if Process.DEBUG_PRINT: print(f"  assigning \"to_check\"={self.name}")
            to_check = self
            
        if len(self)==0:
            return False
        
        if len(self)==1:
            l_p_0=self.linked_process[0]
            if not isinstance(l_p_0, Process): #<--- 'operation' type 
                return False            #but let it pass the process(process) object 
        
        for lp in self: #<--- prechecking everything coherent with allowed types
            if not isinstance(lp,Process): 
                raise TypeError(f"found non-process instance inside {self.name}")
                
        for lp in self:  
            if lp==to_check:   #yield True when one of the linked process is == to to_check process
                if Process.DEBUG_PRINT: print(f"    found recursive state of {self.name}")
                return True
            
        for lp in self:
            # only works with straight recursion
            # nested recursion still send it in a loop if recursion is not in the first element of recursive loops array
            straight_recursion=lp._check_recursion(to_check=to_check) #if to_check is called in a self-loop
            if straight_recursion:
                if Process.DEBUG_PRINT: print(f"    found straight recursion of {self.name}")
                return True
            
#             nested_recursion=lp._check_recursion(to_check=lp) #if there is a nested loop in his linked_pr
#             if nested_recursion:
#                 if Process.DEBUG_PRINT: print(f"    found nested recursion of {self.name}")
#                 return True
        return False   
                
            
    def _retrieve_nested_fn(self, fn_array=list()):
        if len(self)>1:
            for pr in self:
                if not isinstance(pr, Process):
                    raise TypeError(f" process object {self.name} corrupted")
        if self._check_recursion():
            raise RecursionError(f" process object {self.name} is recursive. Can't retrieve nested functions")
                
        for pr in self:
            if not isinstance(pr,Process): 
                #implicitply means (together with integrity check) that this obj is an operation
                return [pr] 
            else:
                fn_array+=pr._retrieve_nested_fn()
        return fn_array
    
################################################################################ instance management 

    #called at every add_process, 
    # since it might happend that the instance changes type 
    # meaning an operation might become a process, then the function it contained must be wrapped in a Process
    def _checkset_selfprocess(self):
        if not hasattr(self, 'name'):
            if Process.DEBUG_PRINT: print(f"  initializer calling check of linked_process ")
        else:
            if Process.DEBUG_PRINT: print(f"checking linked_process of {self.name} ")
        if len(self)>1:
            for k, l_p in enumerate(self):
                if not isinstance(l_p, Process): #<---- can only happend when adding a process to current type operation
                    assert callable(l_p)
                    if Process.DEBUG_PRINT:
                        print(f' found non-process instance ...')
                        print(f"   required wrapping of function {l_p}")
                    self.linked_process[k]=Process(l_p)

                        
    #called at every add_process, 
    # since it might happend that the instance changes type                    
    def _checkset_typestatus(self):
        if not hasattr(self, 'name'):
            if Process.DEBUG_PRINT: print(f"  initializer calling check of type ")
        else:
            if Process.DEBUG_PRINT: print(f"checking name and type of {self.name} ")
            
        #-------- type checkset
        older_type=self.type
        N_tot=len(self)
        if not len(self): 
            self.type='uninitialized'
        if len(self)==1:
            if not isinstance(self.linked_process[0],Process):
                self.type='operation'
            else:
                self.type='process'
        if len(self)>1:
            self.type='process'
        
        #------- name checkset (only changes name when passing from unint-->init, operation-->process)
        if not hasattr(self, 'name'):
            pass  #means the object is being initialized
        elif older_type!= self.type and self._name_status=='auto_named':
            self.name=Process._setget_auto_name(self)
      
    #ensures the instance is built correctly and everything is coordinated
    #called at the end of every add_process and set_name and initialization (through add_process)
    def _checkset_selfstatus(self): 
        if not hasattr(self, 'name'):
            if Process.DEBUG_PRINT: print(f"  initializer calling check of self integrity ")
        else:
            if Process.DEBUG_PRINT: print(f"checking integrity of {self.name} ")
        
        #ensures the instance is built correctly
        assert isinstance(self.linked_process, (tuple,list))
        if len(self)>1:
            for l_p in self:
                if not isinstance(l_p, Process):
                    raise TypeError(f"{self.name} of type {self.type} contains a non Process instance")
        
        
        #type and name checks
        self._checkset_typestatus()
        
                
        #-------- class coordination checkset
        Process._checkset_classtatus()
        
################################################################################ calling

    def _get_internal_input(self, operation, actor):
        internal_inputs_names=operation.__code__.co_varnames[:operation.__code__.co_argcount]  
        # works with args (not kwargs)
        internal_inputs=list()
        for inp_name in internal_inputs_names:
            internal_inputs.append(getattr(actor, inp_name))
        return internal_inputs
    
    def _set_internal_output(self, values, actor):
        if values is not None:
            setattr(actor, self.name, values)
        
    def _operation_execution(self, operation, actor, **external_inputs):
        
        if self.MODE=='implicit':
            # get operation inputs
            internal_inputs=self._get_internal_input(operation, actor)
            # execute operation
            return_values=operation(*internal_inputs, **external_inputs) #<------ PROCESS CORE
            # set operation outputs
            self._set_internal_output(return_values, actor)
            
        elif self.MODE=='explicit':
            operation(actor, **external_inputs)

    def __call__(self, actor, **external_inputs):
        #<----TBI triggers and blockers
        
        #<---- TBI get_input for lenght process?
        for linked_process in self.linked_process: 
            if isinstance(linked_process, Process):
                linked_process(actor, **external_inputs) #<---- the instance of process class knows what to do
            else:
#                 linked_process(actor, **external_inputs) #<------ works only when function is defined as (actor, **kw)
                self._operation_execution(linked_process, actor, **external_inputs)
        #<---- TBI set_output for lenght process?

################################################################################ initialization 
    def __init__(self, *linked_process, name=None, sub_name=None, mode=None, **metadata):
        if Process.DEBUG_PRINT: print("#"*50+"\n !initializer call\n")
            
        self.type='uninitialized'
        self.__control_id=len(Process._all_obj)
        
        self.__init_process(*linked_process)
        self.__init_mode(mode)
#         self._checkset_selfstatus()
        
        self.metadata={**metadata}
        self.__init_name(name, sub_name, **metadata) 
        

        Process.__add_new_instance(self)
        
        if Process.DEBUG_PRINT: 
            print(f"\n intialized with {self.name=} {self.type=}")
            print("#"*50)
        
        
    def __init_process(self, *linked_process):
        if Process.DEBUG_PRINT: print(f" __init__process call ")
        assert not hasattr(self, 'linked_process')
        self.linked_process=[]
        self.add_process(*linked_process)
        if Process.DEBUG_PRINT: print(f" __init__process done\n")
            
    def __init_mode(self, mode):
        if Process.DEBUG_PRINT: print(f" __init_mode call ")
        assert not hasattr(self, 'mode')
        if mode is None:
            self.mode=Process.MODE
        else:
            if mode not in Process._allowed_mode:
                raise ValueError(f"unknown mode {mode}"+f". Available are {Process._allowed_mode}")
        if Process.DEBUG_PRINT: print(f" __init_mode done \n")

    def __init_name(self, name, sub_name, **metadata):
        if Process.DEBUG_PRINT: print(f" __init_name call ")
        assert not hasattr(self, 'name')
        if name is None:
            self._name_status='auto_named'
            if len(self)==1: #works when the function provided is method or lambda
                l_p=self.linked_process[0]
                
                if not isinstance(l_p, Process):
                    function_name=l_p.__code__.co_name #<----try to retrieve name and subname from function name
                    key_sep_name=function_name.split(Process._key_sep)
                    if len(key_sep_name)>=1:
                        name=key_sep_name[0]
                    if len(key_sep_name)>=2:  
                        #<---- if a key_indicaor is found in function name, takes the following as subname
                        self._name_status='user_named'
                        sub_name=key_sep_name[1]
                    if len(key_sep_name)>=3:
                        #<---- if more key indicatore are found, takes the rest as metadata
                        for k, sub_sub_name in enumerate(key_sep_name[2:]):
                            self.metadata['sub_name_'+str(k+2)]=sub_sub_name 
                      
                    # by this time if the object is Operation, name!=None
                    #if sub_name is given as kwarg in initiailization, 
                    #      can be overwritten by key_indicator function naming
                    
                else:
                    name=Process._setget_auto_name(self)       
            else:
                name=Process._setget_auto_name(self)
        else:
            self._name_status='user_named'
            
                #preference for subname is:
        #    - explicitly given as sub_name
        #    - in function name after key_indicator
        #    - taken by initialization kwarg 'subname'
        #    - None
        if sub_name is None:
            sub_name=metadata.get('subname',None)
            if sub_name is not None:
                self.metadata.pop('subname')

        if not isinstance(name, str):
            raise TypeError("provided initialization name must be None or a string")    
        self.name=name
        
        if not (isinstance(sub_name, str) or sub_name is None):
            raise TypeError(f"unable to set sub_name with provided type: {type(subname)=}")
        self.sub_name=sub_name
        # initialization calls __add_new_instance after this 
        if Process.DEBUG_PRINT: print(f" __init_name done \n")

################################################################################ user interface 

    def add_process(self, *multiple_linked_process):
        if not hasattr(self, 'name'): #<--- means it's the initializer call
            if Process.DEBUG_PRINT: print("-"*50+"initializer calling add_process")
        else:    
            if Process.DEBUG_PRINT: print("-"*50+f"add_process call by {self.name}")
        if Process.DEBUG_PRINT: print(f"  trying to add {multiple_linked_process}")
        for linked_process in multiple_linked_process:
            if isinstance(linked_process, dict): #check unallowed types
                raise ValueError(f"can't take a {type(linked_process)} to initialize Process")

            if isinstance(linked_process, (tuple, list)): 
                for l_p in linked_process:
                    self.add_process(l_p)  #recursively unpack every iterable and nested iterable
            elif linked_process is None:
                pass
            else:
                if not callable(linked_process):
                    raise TypeError(" add_process argument(s) must be callable of iterable of callable."
                                    +f" type of passed argument is {type(linked_proces)}")
                self.linked_process.append(linked_process)
        
        #checkset if the instance subprocess are build correctly
        self._checkset_selfprocess()
        
        #checkset if there has been a change in his type
        self._checkset_typestatus()
        
        #check the whole instance is built correctly and rechecks everything
        self._checkset_selfstatus()
        if Process.DEBUG_PRINT: 
            print(f" add_process done")            
            print("-"*50+"\n")
            
    def set_name(self, name):
        if Process.DEBUG_PRINT: print(f" set_name call by {self.name}")

#         name_checked=Process._checkget_name(name)  #<---if I want to print some warning at the most
        if not isinstance(name, str):
            raise TypeError("provided name must be a string")
            
        self._name_status='user_named'
        self.name=name
        
        #check the whole instance is built correctly and rechecks everything
        self._checkset_selfstatus()
        
    def set_mode(self, mode):
        if Process.DEBUG_PRINT: print(f" set_mode call by {self.name}")
            
        if mode.lower() in ['im','imp', 'impl', 'id', 'id_calc', 'implicit', 'id_calculus']:
            mode='implicit'
        elif mode.lower() in ['ex','exp','expl', 'fn', 'fn_calc', 'explicit', 'fn_calculus']:
            mode='explicit'
            
        if mode not in Process._allowed_mode:
            print(f"Unknown mode {mode}. Choose one in {Process._allowed_mode}")
            return
        
        self.mode=mode
    
    #returns the objects in his linked_process array
    #  without any recursivity and type checking
    #returns an array in any case
    def get_linked_process(self):
        if not len(self):
#             print(f"process {self.name} is linked to no process: status {self.type}")
            return []
        if len(self)==1:
            return self.linked_process#[0]
        return self.linked_process
    
    #returns the function nested in his linked_process
    #  recursively retrieving them from the linked_process
    #returns an array in any case
    def get_all_functions(self):
        return self._retrieve_nested_fn()

# pr=Process()

    
PROGRAM_MODE='simulation'    
               
def get_program_mode():
    print(f" current {PROGRAM_MODE=}")           
    return PROGRAM_MODE

def set_program_mode(mode):
    print(f" current {PROGRAM_MODE=}")          
    if 'exo' in mode.lower():
        PROGRAM_MODE='exogenous'
        print(f"program mode set to {PROGRAM_MODE}")
        default_parameter['decision']['price_fluct']=0.
        print(f"  allowed price fluctuations set to {default_parameter['decision']['price_fluct']}")
    elif 'sim' in mode.lower():
        PROGRAM_MODE='simulation'
        print(f"program mode set to {PROGRAM_MODE}")
        default_parameter['decision']['price_fluct']=0.001
        print(f"  allowed price fluctuations set to {default_parameter['decision']['price_fluct']}")
    elif 'strat' in mode.lower():
        print("NOT IMPLEMENTED")
    else:
        print("UNKNOWN PROGRAM MODE")
        print("choose a program mode from 'simulation', 'exogenous', 'strategic'")
    
                     
# __da fare__

# - includi oggetto Broker

# - metric.network come instance di networkx + meccanismi degli agenti di influenzarsi l'un l'altro

# 
# - più partizioni dello stesso mercato
# 
# - più beni d'acquisto (classe stock che eredita da book?)

# COSE CHE DEVO RICORDARE

# 
# - le funzioni del book operano per quantità di stock (non quantità di moneta) quindi l'agente dovrà richiedere operazioni al book tramite (price, quantity). Nelle funzioni in cui i book processano le richieste di operazioni, vengono fatte conversioni tra (price, quantity) di buyer e di seller perché quella che si stanno scambiando è il valore price*quantity. Penso sia scomodo e potrei farli cambiare le funzioni per operare con liquidità, ma se vado a toccare quelle funzioni devo ricordarmi di questi intrighi
# 
# - durante ogni step vengono valutati tre loop separatamente sugli agenti. In ordine: funzioni preliminari, funzioni di processing del pensiero, funzioni di processing delle azioni. Le funzioni preliminari dell'agente sono ad esempio spostare le sue variabili dell'ultimo step ad una seconda variabile "last_ ... "; le funzioni di processing del pensiero sono ad esempio valutare i nuovi sentiment sulla base dei sentiment passati degli altri agenti (influenza reciproca step by step). La funzione preliminare in questo caso serve a separare le variabili contenenti tutti i sentiment di ieri dai sentiment di oggi, cosicché il sentiment di oggi di ogni agente possa essere valutato solo sul sentiment di ieri degli altri agenti; questo evita comportamenti random e meno predicibili
# 
# - gli agenti agiscono TUTTI (compresi quelli la cui azione è 'nothing'). Ovverosia passano tutti nel loop di market. Le metriche sono calcolate su tutti quelli che agiscono, che in questo caso sono tutti; ma se dovessi cambiare il loop devo ricordarmi di questo fatto ed eventualmente dividere il processo di sentiment e magari prediction fatto prima di decidere chi sono i today actor
# 
# - nessuna funzione deve operare per return, o mi ci confondo come non mai: ogni funzione modifica i parametri interni dell'agente, o tuttalpiù chiama una funzione di un secondo agente che modifica i suoi parametri interni
# 
# - per l'uso front-end delle funzioni di market guarda documentazione in seguito
# 
# - la conoscenza di cosa sta succedendo ogni giorno è racchiusa in array di dizionari per le transazioni avvenute:
# agent ha transaction_today, transaction_history. La prima è ripulita ad ogni inizio di step ed aggiornata ogni volta che esegue una transazione o si realizza una sua transazione pending.
# Book ha last transaction che contiene un array di transazioni relative all'ultimo agente che ha agito. transaction_today e transaction_history si comportano come ragionevole. Market per sapere cos'è successo deve andarlo a vedere nel suo self.book e nei suoi self.agents

# 
# 
#at step 0 the sentiment is set at 'sent_value'
#should be the last in the process
# def sentiment_start_at_value(sent_value, operation=None):
#     if operation is None:
#         operation=__coord_agent_op 
#     agent_op=_kw_to_agent_op('sentiment', operation) 
#     def routine(agent, value, **kw):
#         if kw['time']==0:
#             agent_op(agent,value)
#             return value
#         return 0.
#     subprocess_fn= lambda a, **kw: routine(a, sent_value, **kw)
#     return Subprocess(subprocess_fn, name='start_at_value', process='sentiment',
#                       operation=operation, sent_value=sent_value)      


debug=False
debug_warn=False

stochastic_mode='mult'

default_parameter={}

#-------------------------------------------------------------------------- default preliminar process
default_parameter['preliminar']={}

def preliminar_value_shifting():
    def routine(self):
        self.state['last_sentiment']=deepcopy(self.state['sentiment']) #deepcopy to include cases where sentiment is an array of values
        return "step value shifting"
    subprocess_fn=lambda ag,**kw: routine(ag)
    return Operation(subprocess_fn, sub_name='value_shifting', name='preliminar', mode='explicit')

def preliminar_nothing():
    def routine(self):
        return 'nothing'
    subprocess_fn=lambda ag,**kw: routine(ag)
    return Operation(subprocess_fn, sub_name='nothing', name='preliminar', mode='explicit')

instantiate_def_preliminar= lambda :preliminar_nothing()
def_preliminar=instantiate_def_preliminar()

#-------------------------------------------------------------------------- default sentiment process
default_parameter['sentiment']={'sentiment_fluct':0.00}

def sentiment_set_zero():
    def routine(self):
        self.state['sentiment']=0.
        return 0.
    subprocess_fn=lambda ag,**kw: routine(ag)
    return Operation(subprocess_fn, sub_name='set_zero', name='sentiment', mode='explicit')

def sentiment_set_noise(sigma='def', avg='def'):
    if sigma == 'def':
        sigma=default_parameter['sentiment']['sentiment_fluct']
    if avg == 'def':
        avg=0
    def routine(self, avg, sigma):
        self.state['sentiment']=np.clip(np.random.normal(avg,sigma),-1.,1.)
        return self.state['sentiment']
    subprocess_fn= lambda ag, **kw: routine(ag, avg, sigma)
    return Operation(subprocess_fn, sub_name='set_noise', name='sentiment', mode='explicit',
                     sigma=sigma)

def instantiate_def_sentiment():
    if default_parameter['sentiment']['sentiment_fluct']==0.:
        return sentiment_set_zero()
    else:
        return sentiment_set_noise()
def_sentiment=instantiate_def_sentiment()

#-------------------------------------------------------------------------- default prediction process
default_parameter['prediction']={}

def prediction_linear():
    def routine(self):
        self.state['prediction']=(self.state['sentiment']+1)/2
        return self.state['prediction']
    subprocess_fn=lambda ag,**kw: routine(ag)
    return Operation(subprocess_fn, sub_name='linear', name='prediction', mode='explicit')

def prediction_linear_avg():
    def routine(self):
        self.state['prediction']=((np.array(self.state['sentiment'])+1)/2).mean()
        return self.state['prediction']
    subprocess_fn=lambda ag,**kw: routine(ag)
    return Operation(subprocess_fn, sub_name='linear_average', name='prediction', mode='explicit')

def prediction_weighted_avg(weights):
    def routine(self):
        sentiment_values=(np.array(self.state['sentiment'])+1)/2
        self.state['prediction']=0
        for v, w in zip(sentiment_values, weights):
            self.state['prediction']+=v*w
        return self.state['prediction']
    subprocess_fn=lambda ag,**kw: routine(ag)
    return Operation(subprocess_fn, sub_name='weighted_average', name='prediction', mode='explicit',
                    weights=weights)

def prediction_linear_taxed(taxation):
    if taxation<1:  #taxation should be provided as a 1-100 percentage
        taxation*=100
    if taxation>100 or taxation<0:
        raise ValueError("taxation must be provided as a 1-100 percentage")      
    taxation/=100
    def routine(self, taxation):
        linear_pred=self.state['prediction']=(self.state['sentiment']+1)/2
        self.state['prediction']=linear_pred*(1-taxation)
    subprocess_fn=lambda ag,**kw: routine(ag, taxation)
    return Operation(subprocess_fn, sub_name='linear_taxed', name='prediction', mode='explicit')

instantiate_def_prediction= lambda : prediction_linear()
def_prediction=instantiate_def_prediction()

#-------------------------------------------------------------------------- default decision process
# allow_mkt_subaction=False
default_parameter['decision']={}

default_parameter['decision']['def_d_max']=0.75
default_parameter['decision']['def_d_min']=0.25
default_parameter['decision']['def_b_min']=0.0 

default_parameter['decision']['sw_mode']='beta'
# default_parameter['decision']['sw_mode']='uniform'

default_parameter['decision']['eta_coeff']=55
default_parameter['decision']['sw_coeff']='eq_dist'
default_parameter['decision']['sw_coeff_minmax']=0.020
default_parameter['decision']['sw_coeff_postmap']=lambda x, **d_kw,: d_kw['sw_coeff_minmax']*(1-x)+(1-d_kw['sw_coeff_minmax']*1)*x

default_parameter['decision']['price_fluct']=0.001
default_parameter['decision']['reference_price']='last'

default_parameter['decision']['subaction_par']=0.
# beta
# default_parameter['decision']['subaction_par']=1.


def def_choice_input(self):
#     return (self.sigma()-self.state['prediction'])*2-1
    return self.state['prediction']*2-1

def def_D(p, **d_kw):
    d_max=d_kw['def_d_max']
    d_min=d_kw['def_d_min']   
    return 1-(d_min+(d_max-d_min)*(1-np.abs(p)**2))
#     return np.ones(len(p)) CASO LINEARE

def def_B(p, **d_kw):
    l_min=(1-d_kw['def_b_min'])
    return l_min*p
#     return p

#for an action (buy/sell) a fixed proportion d_kw['subaction_par'] will execute that action at market price (any price)
# 0 nobody, 1/2 half of them, 1 everybody
def SA_const(p, P, **d_kw):
    return d_kw['subaction_par']
 
# a proportion d_kw['subaction_par'] times the probability of that action will execute that action at market price (any price)
def SA_lin(p, P, **d_kw):
    return P(p)*d_kw['subaction_par']

def_SA=SA_lin

def decision_stop_days(stop_days_array):
    @timer_agent_def_decision
    def routine(self, stop_days_array, **kw):
        if kw['time'] in stop_days_array:
            self.state['decision']='nothing'
        return 'nothing'
    subprocess_fn=lambda ag,**kw: routine(ag, stop_days_array, **kw)
    return Operation(subprocess_fn, sub_name='holidays', name='stop_days', mode='explicit')


def decision_choice_tr_functional_binomial(D_fn, B_fn, SA_fn, **decision_parameters):
    # signature above is more abstract
    # signature below needs to assume the agent HAS the variables (functions) D_f, B_f, SA_f
    # def decision_choice_tr_functional_binomial():
    
    #deepcopy so that parameters used inside routine will not change if original parameter object is changed
    d_kw=deepcopy(decision_parameters)
    D_f=lambda p: D_fn(p, **d_kw)
    B_f=lambda p: B_fn(p, **d_kw)
    SA_f=lambda p, P: SA_fn(p, P, **d_kw)

    @timer_agent_def_decision
    @timer_agent_def_sample_choice
    def routine(self):
        to_evaluate=self.sp_choice_input(self)   #which in default case correspond to sentiment

        prob_something=D_f(to_evaluate)
        #P(Buy|Smt)-P(Sell|smt)=B
        #P(Buy|Smt)+P(Sell|smt)=1 ---> P(S|s)=1-P(B|s)
        #P(B|s)-(1-P(B|s))=B   ----> P(B|s)=(B+1)/2
        prob_buy_condt=(B_f(to_evaluate)+1)/2

        if np.abs(to_evaluate)>1:
            print("ERROR: to_evaluate=",to_evaluate)

        if np.random.binomial(1, prob_something):
            if np.random.binomial(1, prob_buy_condt):
                decision='buy'
            else:
                decision='sell'
        else:
            self.state['decision']='nothing'
            return 'nothing'

#         if allow_mkt_subaction:
    #         print("WARNING: USING A DEPRECATED (NOT CURRENTLY TESTED) VERSION OF THE DEFAULT DECISION PROCESS")
        if decision =='buy' :   #not tested after 2.7
            if np.random.binomial(1, SA_f(to_evaluate, #this is sloppy should rework concept of subaction
                                             lambda _: prob_something*prob_buy_condt)):
                decision+='_mkt'
        if decision =='sell':   #not tested after 2.7
            if np.random.binomial(1, SA_f(to_evaluate, #this is sloppy should rework concept of subaction
                                             lambda _: prob_something*(1-prob_buy_condt))):
                decision+='_mkt'

        self.state['decision']=decision
        return decision
    subprocess_fn=lambda ag,**kw: routine(ag)
    return Operation(subprocess_fn, sub_name='functional_binomial', name='tr_choice', mode='explicit')

instantiate_def_choice_tr= lambda :decision_choice_tr_functional_binomial(def_D, def_B, def_SA, **default_parameter['decision'])
# def_def_choice_tr=instantiate_def_choice_tr()

def decision_price_tr_gaussian_multiplicative(ref_price='def', price_fluct='def'):
    if ref_price == 'def':
        ref_price  =default_parameter['decision']['reference_price']
    if price_fluct=='def':
        price_fluct=default_parameter['decision']['price_fluct']
        
    @timer_agent_def_decision
    @timer_agent_def_sample_price
    def routine(self, ref_price, price_fluct):
        if self.state['decision']=='nothing':
            self.price='not_req'
            return 'not_req'

        #sampling price from gaussian multiplicative
        if 'mkt' not in self.state['decision']:
            mkt_price=self.book.get_last_price(mode=ref_price)
    #         if stochastic_mode=='add': deprecated and should be conceptually incorrect
    #             price=mkt_price+np.random.normal(0,price_fluct)
            if stochastic_mode=='mult':
                rnd_prc=np.random.normal(0, price_fluct)
                if rnd_prc>=0.: price=mkt_price*(1+rnd_prc)
                if rnd_prc<0.:  price=mkt_price/(1-rnd_prc)

        #price if subaction is 'mkt' (agent decides to buy at market price (any price))
        if 'mkt' in self.state['decision']:
            if 'buy' in self.state['decision']:
                price=self.book.get_worst_sell_offer()
            if 'sell' in self.state['decision']:
                price=self.book.get_worst_buy_offer()

        self.price=price
        return price
    subprocess_fn=lambda ag,**kw: routine(ag, ref_price, price_fluct)
    return Operation(subprocess_fn, sub_name='gaussian_multiplicative', name='tr_price', mode='explicit', 
                      ref=ref_price, sigma=price_fluct )

def decision_price_tr_onesided_gaussian_multiplicative(ref_price='def', price_fluct='def'):
    if ref_price == 'def':
        ref_price  =default_parameter['decision']['reference_price']
    if price_fluct=='def':
        price_fluct=default_parameter['decision']['price_fluct']
        
    @timer_agent_def_decision
    @timer_agent_def_sample_price
    def routine(self, ref_price, price_fluct):
        if self.state['decision']=='nothing':
            self.price='not_req'
            return 'not_req'

        #sampling price from gaussian multiplicative
        if 'mkt' not in self.state['decision']:
            mkt_price=self.book.get_last_price(mode=ref_price)
    #         if stochastic_mode=='add': deprecated and should be conceptually incorrect
    #             price=mkt_price+np.random.normal(0,price_fluct)
            if stochastic_mode=='mult':
                rnd_prc=np.random.normal(0, price_fluct)
                ####################################### code for ONESIDED
                if 'buy' in self.state['decision']:
                    rnd_prc=-np.abs(rnd_prc)
                if 'sell' in self.state['decision']:
                    rnd_prc= np.abs(rnd_prc)
                #######################################
                if rnd_prc>=0.: price=mkt_price*(1+rnd_prc)
                if rnd_prc<0.:  price=mkt_price/(1-rnd_prc)

        #price if subaction is 'mkt' (agent decides to buy at market price (any price))
        if 'mkt' in self.state['decision']:
            if 'buy' in self.state['decision']:
                price=self.book.get_worst_sell_offer()
            if 'sell' in self.state['decision']:
                price=self.book.get_worst_buy_offer()

        self.price=price
        return price
    subprocess_fn=lambda ag,**kw: routine(ag, ref_price, price_fluct)
    return Operation(subprocess_fn, sub_name='onesided_gaussian_multiplicative', name='tr_price', mode='explicit', 
                      ref=ref_price, sigma=price_fluct )
instantiate_def_price_tr=lambda :decision_price_tr_gaussian_multiplicative()
# beta
# instantiate_def_price_tr=lambda :decision_price_tr_onesided_gaussian_multiplicative()
        

def decision_wealth_tr_uniform():
    @timer_agent_def_decision
    @timer_agent_def_sample_wealth
    def routine(self): 
        if self.state['decision']=='nothing':
            self.wealth_tr=0
            self.quantity=0
            return 0
        
        if self.state['decision']=='buy':
            max_q=self.wealth    
        if self.state['decision']=='sell':
            max_q=self.stock*self.price

        #sampling from uniform distribution (does not require any other operation)
        wealth_tr= np.random.uniform(0,max_q)
        self.quantity=wealth_tr/self.price
        self.wealth_tr=wealth_tr
        return wealth_tr  
    
    subprocess_fn=lambda ag,**kw: routine(ag)
    return Operation(subprocess_fn, sub_name='uniform', name='tr_wealth', mode='explicit')
    
#function for sampling a wealth the agent decide to transfer
#in default behaviour is uniform over a swtr_scale_factor fraction of dominio
#will be assigned to new default agents in self.sample_wealth_tr=...
def decision_wealth_tr_parametric_balanced(sw_coeff='def', sw_coeff_postmap='def', sw_mode='def', eta_coeff='def'):
    #about distribution we are sampling from
    if sw_mode=='def':
        sw_mode=default_parameter['decision']['sw_mode']
    if sw_mode=='beta':
        if eta_coeff=='def':
            eta_coeff=default_parameter['decision']['eta_coeff']
            
    #about parameter of that distribution 
    if sw_coeff=='def':
        sw_coeff=default_parameter['decision']['sw_coeff']
    if sw_coeff_postmap=='def': #this could yield unexpected behaviour if not cared properly
        local_kw_copy=deepcopy(default_parameter['decision'])  #<----------- TO AVOID MUTABLE IN LAMBDA
        sw_coeff_postmap=lambda x: local_kw_copy['sw_coeff_postmap'](x, **local_kw_copy) 

        
    @timer_agent_def_decision
    @timer_agent_def_sample_wealth
    def routine(self, sw_mode, eta_coeff, sw_coeff, sw_coeff_postmap): 
        if self.state['decision']=='nothing':
            self.wealth_tr=0
            self.quantity=0
            return 0

        if 'buy' in self.state['decision']:
            max_q=self.wealth    
        if 'sell' in self.state['decision']:
            max_q=self.stock*self.price

        #sampling from mode complex distributions
        #--determining coefficient of average sampling value
        #--sw_coeff can be a number (which fixes avg_coeff), or a identifyer string
#         sw_coeff=self.decision_kw['sw_coeff']
        if type(sw_coeff)!=str:#from beta with average given value fixed to sw_coeff 
            avg_coeff= sw_coeff #which should be float in this case
        if sw_coeff=='eq_dist':#from beta with average C(|sigma-eq|)
            sw_coeff=np.abs(self.distance_from_eq())
            sw_coeff=sw_coeff_postmap(sw_coeff)
            avg_coeff= sw_coeff
        else:
            print("NOT IMPLEMENTED")

        #--sampling wealth_tr from parametric beta distribution
        if sw_mode=='beta':
            wealth_tr=0
            if max_q>0:
                a=avg_coeff*eta_coeff
                b=(1-avg_coeff)*eta_coeff
                wealth_tr=beta.rvs(a, b)*max_q

        self.wealth_tr=wealth_tr
        # setting quantity as wealt_tr/price
        self.quantity=self.wealth_tr/self.price
        return wealth_tr
    subprocess_fn=lambda ag, **kw: routine(ag, sw_mode, eta_coeff, sw_coeff, sw_coeff_postmap)
    return Operation(subprocess_fn, sub_name='parametric_balance', name='tr_wealth', mode='explicit', 
                     dist=sw_mode, coeff=sw_coeff)

def instantiate_def_wealth_tr():
    if default_parameter['decision']['sw_mode']=='uniform':
        return decision_wealth_tr_uniform()
    elif default_parameter['decision']['sw_mode']=='beta':
        return decision_wealth_tr_parametric_balanced()
# def_wealth_tr=instantiate_def_wealth_tr()

def decision_cut_too_small_tr():
    
    @timer_agent_def_decision
    def routine(self): #this ext_kwargs not used in default behaviour

        if self.state['decision']!='nothing' and self.wealth_tr<0.01:
            self.state['decision']='nothing'
            self.quantity=0.
            self.wealth_tr=0.
            self.price='not_req'
            return 'cut'
            
        return 'pass'
    subprocess_fn=lambda ag, **kw: routine(ag)       
    return Operation(subprocess_fn, sub_name='cut_too_small', name='tr_normalize', mode='explicit')

instantiate_def_decision_normalizer= lambda: decision_cut_too_small_tr()
# def_decision_normalizer=instantiate_def_decision_normalizer()

def instantiate_def_decision():
    decision_subprocess_array=[instantiate_def_choice_tr(), 
                               instantiate_def_price_tr(), 
                               instantiate_def_wealth_tr(), 
                               instantiate_def_decision_normalizer()
                              ]
    return decision_subprocess_array
def_decision=instantiate_def_decision()

#-------------------------------------------------------------------------- default action process
default_parameter['action']={}

def action_call_book():

    @timer_agent_def_action
    def routine(self):

        if debug:
            print("agent",self.id,"decision:", self.state['decision'])
            print("agent current wealth:", self.wealth,"; current stock:", self.stock)
            if self.state['decision']!='nothing':
                print("willing at price", self.price, "for quantity:", self.quantity) 
        if self.state['decision']=='nothing':
            return 'nothing'
        elif 'buy' in self.state['decision']:
            order_type='buy'
        elif 'sell' in self.state['decision']:
            order_type='sell'

        if self.quantity>0:  #this check should always be true (check is done in decision)
            self.book.process_new_order(order_type, self.price, self.quantity , self)
            return 'executed'
        
        return 'nothing'
    
    subprocess_fn=lambda ag, **kw: routine(ag)
    return Operation(subprocess_fn, sub_name='call_book', name='action_execution', mode='explicit')
instantiate_def_action=lambda : action_call_book()    
def_action=instantiate_def_action()

#----------------------------------------------------------------------------------- interface with default processes

def _re_instantiate_default_fn():
    global def_preliminar
    global def_sentiment
    global def_prediction
    global def_decision
    global def_action
    def_preliminar=instantiate_def_preliminar()
    def_sentiment =instantiate_def_sentiment()
    def_prediction=instantiate_def_prediction()
    def_decision  =instantiate_def_decision()
    def_action    =instantiate_def_action()    

def change_default_parameter(process, parameter, value):
    global default_parameter
    default_parameter[process][parameter]=value
    
    _re_instantiate_default_fn()
    
def change_default_process(process, function):
    if process=='preliminar':
        global instantiate_def_preliminar
        instantiate_def_preliminar=lambda :function
    if process=='sentiment':
        global instantiate_def_sentiment
        instantiate_def_sentiment=lambda :function
    if process=='prediction':
        global instantiate_def_prediction
        instantiate_def_prediction=lambda :function
    if process=='decision':
        global instantiate_def_decision
        instantiate_def_decision=lambda :function
    if process=='action':
        global instantiate_def_action
        instantiate_def_action=lambda :function
    _re_instantiate_default_fn()

def activate_beta():
    global default_parameter
    default_parameter['decision']['subaction_par']=1.
    
    global instantiate_def_price_tr
    instantiate_def_price_tr=lambda : decision_price_tr_onesided_gaussian_multiplicative()

    _re_instantiate_default_fn()

class Qecon:
        
    def __repr__(self):
        dir_info=" Agent "+str(self.id)+" of partition "+str(self.group)
        dir_info+="\n sigma {:.4f}  tot wealth: {:.2f}".format(self.sigma(), self.tot_wealth())
#         print(dir_info)
        return dir_info
   
    def __init__(self, idx=None, kind='def', partition='gen', wealth=5000, stock=50, 
                 keep_tr_memory=False, keep_all_tr_memory=False):
        self.id=idx
        self.group=partition
        
        self.wealth=wealth
        self.stock=stock
        self.price=0.    # decided price for current transaction
        self.quantity=0. # decided quantity for current transaction
        
        self.book=None 
#         self.executed=False
        self.traded_quantity=0.
        self.transaction_today=[]
        self.transaction_history=[]
        self.keep_tr_memory=keep_tr_memory  #if true it will allow to fill tr_today and tr_history
        self.keep_all_tr_memory=keep_all_tr_memory #if true it will allow to fill tr_history
        
#         self.sentiment=0.
#         self.prediction=0.
#         self.decision='nothing'
#         self.last_sentiment=0.
        #can be generalized to all his internal variables if needed

        self.state={}
        self.process={}
        self.parameter={}
        
        self.state['sentiment']=0.
        self.state['last_sentiment']=0.
        self.state['prediction']=0.5
        self.state['decision']='nothing'
        
        if kind=='def':
            #defined inside 'kind' for no lack of generality
            #every set_process(name,...) silently initialize that process
            self.set_process('preliminar', def_preliminar)
            self.set_process('sentiment',  def_sentiment)
            self.set_process('prediction', def_prediction)
            self.set_process('decision',   def_decision)
            self.set_process('action',     def_action)
            
            # needed for evaluating distance from equilibrium
            #do not like it but best solution (deepcopy ensure this is a snapshot of current B ansatz)
                #ERROR WHEN AN AGENT CHANGES ITS B ANSATZ AFTER DEFINITION
            self.my_B=lambda p: def_B(p, **deepcopy(default_parameter['decision'])  )

            self.sp_choice_input=def_choice_input
    
        elif kind=='noise':
            #defined inside 'kind' for no lack of generality
            #every set_process(name,...) silently initialize that process
            self.set_process('preliminar', preliminar_nothing() )
            self.set_process('sentiment',  sentiment_start_at_value(0) )
            self.set_process('prediction', def_prediction)
            self.set_process('decision',   def_decision)
            self.set_process('action',     def_action)
            
            # needed for evaluating distance from equilibrium
            #do not like it but best solution (deepcopy ensure this is a snapshot of current B ansatz)
                #ERROR WHEN AN AGENT CHANGES ITS B ANSATZ AFTER DEFINITION
            self.my_B=lambda p: def_B(p, **deepcopy(default_parameter['decision']) )
            
            self.sp_choice_input=lambda *args, **kwargs: 0 
        
#         if kind=='empty': this code is deprecated in 2.10; also never used empty agents
#             self.my_pers['sentiment'] =[null_kwfn]
#             self.my_pers['prediction']=[null_kwfn]
#             self.my_pers['decision']  =[null_kwfn]
#             self.my_pers['action']    =[null_kwfn]
        else:
            raise ValueError(f"Unknown agent kind {kind}")

    #every agent has a dictionary of functions 
    #  representing his way of processing information (process['sentiment'] etc )
    #every agent stores internally the results of these process
    #  represent the "agent knows what he is doing" principle
    #agent function never call each other but looks at stored processed info
    #  for ease of programming reason
    #processes are divided in PRELIMINAR, THOUGHT, ACTION
    #  preliminar is executed before beginning of the step
    #  thought and action are divided for "conceptual" understanding
    #  broadly, thought should only modify agent state, action eveutually external variables
    #there is thus a hierarchy like: 
    #  superprocess (preliminar, thought, action)
    #  process(sentiment, prediction, decision, action)
    #  operations(the way each one of these is executed)
    #currently the relation between superprocess and process is fixed but could be generalized
    def clear_process(self, process_name):
        self.process[process_name]=[]
    
    @timer_agent_tot_preliminar
    def process_preliminar(self, **ext_args):
        for preliminar_process in self.process['preliminar']:
            preliminar_process(self,**ext_args)
            
    @timer_agent_tot_thought    
    def process_thought(self, **ext_args):
#         self.executed=False
        for sentiment_process in self.process['sentiment']:
            sentiment_process(self,**ext_args)  
        for prediction_process in self.process['prediction']:
            prediction_process(self,**ext_args)
            
    @timer_agent_tot_action    
    def process_action(self,**ext_args):
#         self.executed=False
        for decision_process in self.process['decision']:
            decision_process(self,**ext_args)
        for action_process in self.process['action']:
            action_process(self,**ext_args)
            
    def summary(self):#, show_last_result=True):
        print()
        print('#'*70)
        print(self)
        for process, subprocess_array in self.process.items():
            print()
            print(" process {:_<60}".format(process))
            for subprocess in subprocess_array:
                print(subprocess)
#                 if show_last_result:
#                     if isinstance(subprocess, Subprocess_Class):
#                         print("       last result:", subprocess.last_result)
        
            
    def set_process(self, process_name, subprocess): 
        if process_name not in self.process:
            self.process[process_name]=[] #<--- silently initializing process
            
        if isinstance(subprocess, (list,tuple)):
            self.process[process_name]=[*subprocess]
        else:
            self.process[process_name]=[subprocess]
        
    def add_process(self, process_name, subprocess):
        if isinstance(subprocess, (list,tuple)):
            for subpr in subprocess:
                self.process[process_name].append(subpr)
        else:
            self.process[process_name].append(subprocess)
        
    def add_sentiment_process(self, fn):
        self.add_process('sentiment', fn)
        
    def process_sale(self, price, quantity):
#         self.executed=True
        self.traded_quantity-=quantity
        self.stock-=quantity
        self.wealth+=price*quantity
        if self.stock<0 :
            if debug_warn: print("   WARNING agent variable <0: self.stock:",self.stock)
            if self.stock>-1e-7:    #non mi piace ma è il modo più semplice per gestire
                self.stock=0.                       #errori numerici del pc di rounding 
        if self.wealth<0:
            if debug_warn: print("   WARNING agent variable <0: self.wealth:",self.wealth)
            if self.wealth>-1e-7:  #gli sto dando una tolleranza di 10^-7 
                self.wealth=0.
        if debug:
            print("   agent", self.id," new wealth:",self.wealth,"; new stock",self.stock )
            assert self.stock>=0
            assert self.wealth>=0
            
    def process_buy(self, price, quantity):
#         self.executed=True
        self.traded_quantity+=quantity
        self.stock+=quantity
        self.wealth-=price*quantity
        if self.stock<0 :
            if debug_warn: print("   WARNING agent variable <0: self.stock:",self.stock)
            if self.stock>-1e-7:
                self.stock=0.
        if self.wealth<0:
            if debug_warn: print("   WARNING agent variable <0: self.wealth:",self.wealth)
            if self.wealth>-1e-7:
                self.wealth=0.
        if debug:
            print("   agent", self.id," new wealth:",self.wealth,"; new stock",self.stock )
            assert self.stock>=0
            assert self.wealth>=0
            
    def add_transaction_history(self, transaction):
        if self.keep_tr_memory:
            self.transaction_today.append(transaction)
            if self.keep_all_tr_memory:
                self.transaction_history.append(transaction)
        
        
    def tot_wealth(self, price=None):
        if price==None:
            price=self.book.get_last_price(mode='last')
        tot_wealth=self.stock*price+self.wealth
        return tot_wealth
             
    def sigma(self, tr='prop', price=None):
        if price==None:
            price=self.book.get_last_price(mode='last')
        if tr=='prop':
            sigma=self.stock*price/(self.stock*price+self.wealth)
        elif tr=='diff':
            sigma=(self.stock*price-self.wealth)/(self.stock*price+self.wealth)
        return sigma
    
    def equilibrium(self, tr='prop', price=None):
        if price==None:
            price=self.book.get_last_price(mode='last')
        my_b=self.my_B(self.sp_choice_input(self))
#         my_b=self.B_f(self.state['prediction']*2-1, **self.decision_kw)
        if tr=='diff':                            #SKETCHY non mi piace ci siano due modi per vedere sigma
            return my_b                        #sono uno la trasformazione lineare dell'altro da -1,1 a 0,1
        elif tr=='prop':                             #vedo qual è più interpretabile (credo prop) e tengo solo quello
            return (my_b+1)/2
         
    def distance_from_eq(self, tr='prop', price=None):
        my_sigma=self.sigma(tr, price)
        exp_equi=self.equilibrium(tr, price)
        return my_sigma-exp_equi
                
    def take_parameters(self, parameters):
        if parameters==None:
            return
        for parname, parval in parameters.items():
#             if argname in parameter:
#                 print("WARNING ", argname," already defined")
            self.parameter[parname]=parval
    
    def clear(self):
        try:
            assert self.stock>=0
            assert self.wealth>=0
        except:
            print("WARNING value error:")
            print("agent", self.id,"stock:", self.stock)
            print("agent", self.id,"wealth:", self.wealth)
                  
#         self.executed=False
        self.state['decision']='nothing'
        self.price=None
        self.traded_quantity=0.
        self.transaction_today=[]
        
    def afterwarmup_clear(self):
        self.state['sentiment']=0.
        self.state['prediction']=0.
        self.state['decision']='nothing'
        self.price=0.
        self.quantity=0.
#         self.executed=False
        self.traded_quantity=0.
        self.transaction_today=[]
        self.transaction_history=[]

    
class Book:

    def __init__(self, start_price=100., program_mode=None, exogenous_price=None):
        if program_mode is None:
            program_mode=PROGRAM_MODE    

        if exogenous_price is not None and program_mode != 'exogenous':
            print("provided exogenous price of size", len(exogenous_price))
            program_mode='exogenous'
            print("book instance program mode will be set to exogenous")
        self.program_mode=program_mode
                  
        if self.program_mode=='exogenous':
            try:
                start_price=exogenous_price[0]
                self.set_exogenous_price(exogenous_price)
            except:
                print(" program mode is set to", self.program_mode)
                print(" provide price data through set_exogenous_price( iterable of daily prices )")
            self.time=0
        if self.program_mode=='simulation':
            self.time=0  #<----2.7.0 not using it in this program_mode
#             no specific operation
#         if self.mode=='strategic':
#             not implemented
            
        self.last_price=start_price
        self.open_price=self.last_price
        
        self.buy_order=[]
        self.sell_order=[]

        self.volume=0.
        
        self.market=None
        
        self.transaction_history=[]
        self.transaction_today=[]
#         self.last_transaction=[]
        
    def set_exogenous_price(self, price_array):
        self.exogenous_price=price_array
        print("  simulation will run in mode:", self.program_mode)
        try:
            print("  daily prices data size:", len(self.exogenous_price))
        except:
            print("  WARNING price data must be provided")
            print("  call set_exogenous_price( iterable object of daily prices )")
    
    @timer_book_memory
    def add_transaction_history(self, transaction):
        self.transaction_history.append(transaction)
        self.transaction_today.append(transaction)    #cleared at every call of book.clear
#         self.last_transaction.append(transaction)     #cleared at every call of book.process_new_order
    
    #called every time a transaction is made (in process_buy/sell_order)
    def set_last_price(self, price):
        
        if self.program_mode=='simulation':
            self.last_price=price
            
        if self.program_mode=='exogenous': #reference price is always open price in exogenous mode
            self.last_price=self.open_price  
    
    #called with self.clear(), which is called by market.run at the beginning of each step 
    def set_open_price(self, price):
        
        if self.program_mode=='simulation':
            self.open_price=price
            
        if self.program_mode=='exogenous':#open price is always taken from given array
            self.open_price=self.exogenous_price[self.time-1]    
    
    def get_last_price(self, mode='last'):
        
        if self.program_mode=='simulation':
            if mode=='last':
                return self.last_price
            if mode=='open':
                return self.open_price
            if mode=='day_avg':
                print("not implemented")
                
        if self.program_mode=='exogenous':
            return self.open_price
        return 'error'
    
    def get_best_buy_offer(self):
        if len(self.buy_order)==0:
            return self.last_price
        return self.buy_order[0][0]
    
    def get_best_sell_offer(self):
        if len(self.sell_order)==0:
            return self.last_price
        return self.sell_order[0][0]
    
    #called by agents that are willing to accept any price (subaction '_mkt')
    def get_worst_sell_offer(self):
        if len(self.sell_order)==0:
            return self.last_price
        return self.sell_order[-1][0]
    
    #called by agents that are willing to accept any price (subaction '_mkt')
    def get_worst_buy_offer(self):
        if len(self.buy_order)==0:
            return self.last_price
        return self.buy_order[-1][0]
        
    #appends the order type to respective book, when that transaction needs to remain pending
    @timer_book_order_appending
    def add_buy_order(self, price, quantity, agent):
        #sorted max --> min
        ID=agent.id
        if len(self.buy_order)==0:
            self.buy_order=[[price, quantity, ID]]
            if debug:
                print("appended buy order:", [price, quantity, ID], "position", 0)
            return
        else:
            for k, el in enumerate(self.buy_order):
                if price >= el[0]:
                    self.buy_order.insert(k, [price, quantity, ID])
                    if debug:
                        print("appended buy order:", [price, quantity, ID], "position", k)
                    return
            if debug:
                print("appended buy order:", [price, quantity, ID], "position", k)
            self.buy_order.append([price, quantity, ID])
    
    #appends the order type to respective book, when that transaction needs to remain pending
    @timer_book_order_appending
    def add_sell_order(self, price, quantity, agent):
        #sorted min --> max
        ID=agent.id
        if len(self.sell_order)==0:
            self.sell_order=[[price, quantity, ID]]
            if debug:
                print("appended sell order:", [price, quantity, ID], "position", 0)
            return
        else:
            for k, el in enumerate(self.sell_order):
                if price <= el[0]:
                    self.sell_order.insert(k, [price, quantity, ID])
                    if debug:
                        print("appended sell order:", [price, quantity, ID], "position", k)
                    return
            if debug:
                print("appended sell order:", [price, quantity, ID], "position", k)
            self.sell_order.append([price, quantity, ID])
    
    def add_new_order(self, price, SorB, quantity, agent):
        if SorB=='buy':
            self.add_buy_order(price, quantity, agent)
        elif SorB=='sell':
            self.add_sell_order(price, quantity, agent)
            
    
    def process_buy_order(self, buyer_price, buyer_quantity, buyer):
        if buyer_quantity<=0:
            return
        if debug:
            print("processing buy order")
        if len(self.sell_order)==0:
            self.add_buy_order(buyer_price, buyer_quantity, buyer)
            return

        seller_price=self.sell_order[0][0]
        seller_quantity=self.sell_order[0][1]
        seller_id=self.sell_order[0][2]
        seller=self.market.agents[seller_id]
        buyer_id=buyer.id

        if buyer_price>=seller_price:
            if debug:
                print("looking at pending sell order", self.sell_order[0])
            self.sell_order.pop(0)
            self.set_last_price(seller_price)
            
            buyer_resc_quantity=buyer_quantity/seller_price*buyer_price  #<----- rescaling quantity according to different price
            transaction_quantity=min(buyer_resc_quantity, seller_quantity)    
            self.volume+=transaction_quantity
            self.volume_money+=transaction_quantity*seller_price
            
            transaction={
                'b_id':buyer_id,'s_id':seller_id,'price':seller_price, 'quantity':transaction_quantity
            }

            if debug:
                print("###### order executed: sold at price", seller_price)
                print("       transaction:", transaction) 
            buyer.process_buy(seller_price, transaction_quantity)
            seller.process_sale(seller_price, transaction_quantity)
            buyer.add_transaction_history(transaction)
            seller.add_transaction_history(transaction)
            self.add_transaction_history(transaction)
            self.set_last_price(seller_price)
                       
            if buyer_resc_quantity>seller_quantity:
#                 seller_resc_quantity=seller_quantity#/seller_price*buyer_price
                self.process_buy_order(buyer_price, buyer_quantity-transaction_quantity, buyer)
            elif buyer_resc_quantity<seller_quantity:
                self.add_sell_order(seller_price, seller_quantity-transaction_quantity, seller)

        else:
            self.add_buy_order(buyer_price, buyer_quantity, buyer)

            
    def process_sell_order(self, seller_price, seller_quantity, seller):
        if seller_quantity<=0:
            return
        if debug:
            print("processing sell order")
            
        if len(self.buy_order)==0:
            self.add_sell_order(seller_price, seller_quantity, seller)
            return

        buyer_price=self.buy_order[0][0]
        buyer_quantity=self.buy_order[0][1]
        buyer_id=self.buy_order[0][2]
        buyer=self.market.agents[buyer_id]
        seller_id=seller.id
        
        if seller_price<=buyer_price:
            if debug: 
                print("looking at pending buy order", self.buy_order[0])
            self.buy_order.pop(0)
            self.set_last_price(buyer_price)
            
#             seller_resc_quantity=seller_quantity#*seller_price/buyer_price
            transaction_quantity=min(buyer_quantity, seller_quantity)    
            self.volume+=transaction_quantity
            self.volume_money+=transaction_quantity*buyer_price
            
            transaction={
                'b_id':buyer_id,'s_id':seller_id,'price':buyer_price, 'quantity':transaction_quantity
            }

            if debug:
                print("###### order executed: sold at price", buyer_price)
                print("       transaction:", transaction) 
            seller.process_sale(buyer_price, transaction_quantity)
            buyer.process_buy(buyer_price, transaction_quantity)
            seller.add_transaction_history(transaction) 
            buyer.add_transaction_history(transaction)
            self.add_transaction_history(transaction)
            self.set_last_price(buyer_price)
                        
            if seller_quantity>buyer_quantity:
#                 buyer_resc_quantity=buyer_quantity/buyer_price*seller_price
                self.process_sell_order(seller_price, seller_quantity-transaction_quantity, seller)
            elif seller_quantity<buyer_quantity:
                self.add_buy_order(buyer_price, buyer_quantity-transaction_quantity, buyer)

        else:
            self.add_sell_order(seller_price, seller_quantity, seller)
            
    @timer_book_management        
    def process_new_order(self, SorB, price, quantity, agent):
        if quantity<=0:
            return
#         self.last_transaction=[]
        if SorB=='buy':
            self.process_buy_order(price, quantity, agent)
        elif SorB=='sell':
            self.process_sell_order(price, quantity, agent)
        
    def get_book_depth(self,mode='diff'):
        if mode=='diff':
            return len(self.buy_order)-len(self.sell_order)
        
    def clear(self):  #called iff at the beginning of a new step
        self.time+=1
        self.set_open_price(self.get_last_price(mode='last'))
        self.buy_order.clear()
        self.sell_order.clear()

        self.volume=0.
        self.volume_money=0.
        
        self.transaction_today=[]
        
    def afterwarmup_clear(self):
        self.time=0
        self.set_open_price(self.get_last_price(mode='last'))

        self.volume=0.
        self.volume_money=0.
        
        self.buy_order=[]
        self.sell_order=[]
        
        self.transaction_history=[]
        self.transaction_today=[]
#         self.last_transaction=[]


            
class Market:
  
    def __init__(self, book=None, start_price=100., # N_agents=0, deprecated 
                 start_date=date.today(), network='empty', warmup_steps=0,
                 kind='def', process_preliminar=True, init_func=None, 
                 program_mode=None, exogenous_price=None):
        
        if program_mode is None:
            program_mode=PROGRAM_MODE       

        if exogenous_price is not None and program_mode != 'exogenous':
            print("provided exogenous price of size", len(exogenous_price))
            program_mode='exogenous'
            print("This Market instance program mode will be set to exogenous")
            
        self.program_mode=program_mode
        if book ==None:
            book=Book(start_price=start_price, program_mode=program_mode, exogenous_price=exogenous_price)
        self.book=book
        self.book.market=self

        
        self.agent_partition={}
        self.partition_metric={'avg_sent':{}, 'avg_pred':{}}   
        self.partition_metric['daily_count']= {'buy':{},'sell':{} , 'buy_mkt':{} , 'sell_mkt':{},'nothing':{} }
               
        self.n_agents=0 #this value is updated by self.add_agent
        self.agents={}
#         if N_agents>0:
#             self.add_partition(name='group_0', size=N_agents)
#         self.agents={i: Qecon(Id=i, kind=kind) for i in range(N_agents)}
#         self.agent_partition={'gen':{ag.id:ag for ag in self.agents.values()}}

        self.start_date=start_date
        self.date=start_date

        self.time=0
        
        self.set_network(network)
        self.process_preliminar=process_preliminar
        
        self.special_fn={'pre': null_kwfn,'post': null_kwfn}
        
        #to ease up computation (some data can be computed once, instead of re-evaluated for every agent) 
        self.shared_among_agents={} #data passed to agents, when all agents needs to compute the same info
        self.compute_shared_info={} #dictionary of function returning info that will fill shared_among_agents[name]
              
        #every agents contains his id (number)
        #agents are stored in self.agents in form of {id: agent}
        #partition group agent (mutually) in form of
        #dictionary where keys are partition names
        #every agent_partition{partition name: contains {id: agent}
        
        self.price_history=[]
        
        self.metric={}
        self.metric['daily_price']={'open':[], 'close':[], 'avg':[],'min':[], 'max':[]} 
        self.metric['daily_volume']=[]
        self.metric['daily_volume_money']=[]
        self.metric['daily_bookdepth']=[]
        self.metric['daily_count']={'buy':[], 'sell':[], 'buy_mkt':[], 'sell_mkt':[], 'nothing':[]}
        self._default_metric=['daily_price','daily_volume','daily_volume_money','daily_bookdepth','daily_count']
        #some metrics are evaluated by default, some can be added through market.add_metric(...)
        self.metric_start_step={}
        self.metric_inside_step={}
        self.metric_post_step={}
        
        self._default_partition_metric=['avg_sent','avg_pred','daily_count']
        self.partition_metric_start_step={}
        self.partition_metric_inside_step={}
        self.partition_metric_post_step={}
        
        self.custom_infograph=[]
        self.info_trend=0
        self.info_mavdiff=0
        
        #only function called BEFORE warmup. Sets initial value for simulation,
        #while all future values are to be assumed as changes in market parameters
        if init_func!=None:
#             self=init_func(self)
            init_func(self)    #credo possa modificare per reference l'oggetto passato
    
        if warmup_steps>0:   #dopo funzione di inizializzazione
            self._warmup(warmup_steps)
            
####################################################################### mode management
            
    def set_exogenous_price(self, price_array):
        self.book.set_exogenous_price(price_array)

####################################################################### PARTITION/AGENTS FUNCTION
    #market can be created empty if no n_agents is specified (or set to 0)

    def add_agent(self, kind='def', sigma_prop=0.5, tot_wealth=10000, partition_name='group_0', idx='auto'):
        
        #estabilishing agent partition
        #if none is specified, it will be assigned to 'gen'
        #if 'gen' is not present, gives back warning and error
        #if no partition is present (first agent) it build first partition named {partition_name}
        if partition_name not in self.agent_partition:
            print(f"WARNING {partition_name} not currently a partition for market")
            if len(self.agent_partition)==0:   #if this is the first agent
                self.add_partition(name=partition_name)  #builds
                print(f"  created first partition {partition_name}")
            else:
                print("  specify an existing partition for proceeding ")
                print("  current partition:", [p_n for p_n in self.agent_partition])
                return
        
        #building agent
        if idx=='auto':
            idx=self.n_agents+1
        assert idx not in self.agents
        
        curr_price=self.book.get_last_price(mode='last')
        assert sigma_prop>=0. and sigma_prop<=1.
        stock=sigma_prop*tot_wealth/curr_price
        money=tot_wealth-stock*curr_price
        new_agent=Qecon(idx=idx, kind=kind, stock=stock, wealth=money, partition=partition_name)
        
        #linking market with this agent
        self.n_agents+=1
        self.agents[idx]=new_agent
        
        #assigning book to this agent
        new_agent.book=self.book
        
        #linking partition with this agent
        self.agent_partition[partition_name][idx]=new_agent
    
    # size is the number of agents in that partition
    # name will be group_#partition if left to name='auto'
    # tot_wealtn and/or sigma_prop can be exact value:
    #    in which case all agent will share that same value
    # otherwise can be set to 'gauss', 'poiss', 'uniform', ...
    #    in which case a sampling is performed with tot_wealth_avg and/or sigma_prop_avg
    # the parameters "parname"_avg are used only if "parname" (tot_wealth and sigma_prop) are a string
    #    which means they are only used if a sampling is requested
    def add_partition(self, size=0, name='auto', kind='def',
                      tot_wealth=10000, sigma_prop=0.5, tw_avg=10000, sp_avg=0.5):
        if name=='auto':
            if 'group_0' in self.agent_partition:
                name='group_'+str(len(self.agent_partition))
            else:
                name='group_'+str(len(self.agent_partition)+1)
        if name in self.agent_partition:
            print(f"WARNING partition {name} already present")
            return
        
        self.agent_partition[name]={}
        self.partition_metric['avg_sent'][name]=[]
        self.partition_metric['avg_pred'][name]=[]
        for action in self.partition_metric['daily_count']:
            self.partition_metric['daily_count'][action][name]=[]
        
        if VERBOSITY_MODE:
            print()
            print("Building partition", name,": ", size,"agents of kind", kind)
            print()
        if not isinstance(tot_wealth, str):
            if VERBOSITY_MODE: print("  all agents tot_wealth will be:", tot_wealth)
        else:
            if VERBOSITY_MODE: print("  agents tot_wealth will be sampled from",tot_wealth,"distribution")
        if not isinstance(sigma_prop, str): 
            if VERBOSITY_MODE: print("  all agents sigma_prop will be:", sigma_prop)
        else:
            if VERBOSITY_MODE: print("  agents sigma_prop will be sampled from",sigma_prop,"distribution")
        #_kw_to_sample is a function that returns a function that sample according to specifications
        #if first argument of _kw_to_sampler is not a string, it returns a function that just returns first argument
        sigma_prop_sample= _kw_to_sampler(sigma_prop, avg=sp_avg, low=0.1, high=2*sp_avg-0.1, sample_size=size)()    
        tot_wealth_sample= _kw_to_sampler(tot_wealth, avg=tw_avg, low=100, high=2*tw_avg-100, sample_size=size)() 
        
        for sp, tw in zip(sigma_prop_sample, tot_wealth_sample) :
            self.add_agent(sigma_prop=sp, tot_wealth=tw, partition_name=name, kind=kind)
            

    def get_part_agents(self, partition_name):
        return  list(self.agent_partition[partition_name].values())
    
    def part_op_summary(self, partition_name):
        print()
        print("{:#^60}".format(" "+partition_name.upper()+" ") )
        repr_agent=self.get_part_agents(partition_name)[0]
        for process, subprocess_array in repr_agent.process.items():
            print()
            print(" process {:_<60}".format(process))
            for subprocess in subprocess_array:
                print(subprocess)
        
    def all_op_summary(self):
        for part in self.agent_partition:
            self.part_op_summary(part)
            
    
    def make_agents_partition(self, partition_size_name):
        print("    WARNING THIS FUNCTION IS DEPRECATED")
        print("        use add_partition instead")
        self.agent_partition={}
        self.partition_metric={'avg_sent':{}, 'avg_pred':{}}   #clears up previous partition info
        self.partition_metric['daily_count']= {'buy':{},'sell':{} , 'buy_mkt':{} , 'sell_mkt':{},'nothing':{} }

        from_n=0
        for name,size in partition_size_name.items():
            if size=='auto':
                size=len(self.agents)-from_n
            if type(size) is float:
                size=int(len(self.agents)*size)
            self.agent_partition[name]={ag.id: ag for ag in self.agents.values()if ag.id>=from_n and ag.id <from_n+size}

            self.partition_metric['avg_sent'][name]=[]
            self.partition_metric['avg_pred'][name]=[]
            for action in self.partition_metric['daily_count']:
                self.partition_metric['daily_count'][action][name]=[]
            
            from_n+=size
            for ag in self.agent_partition[name].values():
                ag.group=name
                
    def clear_part_process(self, partition_name, process_name):
        if partition_name in self.agent_partition:
            for ag in self.agent_partition[partition_name].values():
                ag.clear_process(process_name)
        
                
    #agent in partition_name will have their 'process' defined by fn    
    def set_part_fn(self, partition_name, process, fn):
        if partition_name in self.agent_partition:
            for ag in self.agent_partition[partition_name].values():
                ag.set_process(process,fn)
    
    #all agents in market will have their 'process' defined by fn
    def set_all_agent_fn(self, process, fn):
        for partition_name in self.agent_partition:
            self.set_part_fn(partition_name, process, fn)
             
    #agent in partition_name will have their 'sentiment' defined by fn 
    #if any process is already present it will be overwritten
    def set_part_sentiment(self, partition_name, fn): 
        self.set_part_fn(partition_name, 'sentiment', fn)
    
    #all agents in market will have their 'sentiment' defined by fn
    #if any process is already present it will be overwritten
    def set_all_agent_sentiment(self, fn): 
        self.set_all_agent_fn('sentiment', fn)
    
    #since each process can be composed of multiple subprocesses, it needs a function for adding a process
    def add_part_fn(self, partition_name, process, fn):
        if partition_name in self.agent_partition:
            for ag in self.agent_partition[partition_name].values():
                ag.add_process(process,fn)
    
    #agent in partition_name will add fn to their sentiment processes
    def add_part_sentiment(self, partition_name, fn): 
        self.add_part_fn(partition_name, 'sentiment', fn)
    
    #all agents in market will add fn to their sentiment processes
    def add_all_agent_sentiment(self, fn):
        for partition_name in self.agent_partition:
            self.add_part_sentiment(partition_name, fn)

    def give_part_parameters(self, partition_name, partition_parameters):
        if partition_name in self.agent_partition:
            for ag in self.agent_partition[partition_name].values():
                ag.take_parameters(parameters=partition_parameters)  
    
                    
    # shared_fn must be in signature fn(mkt, **kw) and return some value 
    # that returned value will be assigned at each step to sentiment of {partition_name}
    def _struct_part_sentiment_byshared(self, partition_name, shared_fn, _struct_op,
                                        info_name='auto', math_operation=None, index=None ):
    # name will be info_
        if info_name=='auto':
            info_name='info_'+str(len(self.compute_shared_info)-self.info_trend-self.info_mavdiff)
        if info_name not in self.compute_shared_info:
            self.add_compute_shared_info(info_name, shared_fn)
    #     partition partition_name take that info
    
        if _struct_op=='set':
            if math_operation is None:
                math_operation='set' 
                #<-------- when requesting a (re)setting of process, 
                #          default behaviour should be assignation operation for that sentiment subprocess
                #  meaning since it is the first value to be computed, it also needs to initialize that value
            if math_operation!='set':
                print("WARNING math_operation",math_operation,"while structural operation is",_struct_op) 
            self.set_part_sentiment(partition_name, sentiment_take_from_shared(info_name, math_operation=math_operation, index=index))
            
        if _struct_op=='add':
            if math_operation is None:
                math_operation='add'
                #<-------- when requesting an additional process, 
                #          default behaviour should be addition math_operation for that sentiment subprocess
                #  meaning since it is not the first value to be computed, it should be an additive subprocess 
            self.add_part_sentiment(partition_name, sentiment_take_from_shared(info_name, math_operation=math_operation, index=index)) 
        
                
                
    # trend_basic(market, distance, end=-1, oneperc_sent=None,
    #                 focus='day', mode='abs', apply_norm='tanh'):
    # trend_multi_basic(market, multi_distance=[1,7,14,30,90,180, 365], end=-1, oneperc_sent=None,
    #                  focus='day', mode='abs', weights=None, return_mode='avg', apply_norm='tanh')
    def _struct_part_sentiment_bytrend(self, partition_name, trend_type, _struct_op,
                                       math_operation, index=None, **trend_kw):
    #     choosing trend function
        if trend_type in ['basic', 'B']:
            trend_fn=lambda self, **ext_kw: trend_basic(self,**trend_kw)
        if trend_type in ['multi_basic','MB']:
            trend_fn=lambda self, **ext_kw: trend_multi_basic(self,**trend_kw)

    #     adding trend to marked computed values
        info_name='trend_'+str(self.info_trend)
        self.info_trend+=1
        
        self._struct_part_sentiment_byshared(partition_name, trend_fn, _struct_op,
                                             info_name=info_name,math_operation=math_operation, index=index) 

        
    def _struct_part_sentiment_bymavdiff(self, partition_name, mavdiff_type, _struct_op,
                                     math_operation, index=None, **mavdiff_kw):
    #     choosing trend function
        if mavdiff_type in ['basic', 'B']:
            mavdiff_fn=lambda self, **ext_kw: mavdiff_basic(self,**mavdiff_kw)
        if mavdiff_type in ['multi_basic','MB']:
            mavdiff_fn=lambda self, **ext_kw: mavdiff_multi_basic(self,**mavdiff_kw)

        info_name='mav_'+str(self.info_mavdiff)
        self.info_mavdiff+=1
        
        self._struct_part_sentiment_byshared(partition_name, mavdiff_fn, _struct_op,
                                             info_name=info_name, math_operation=math_operation, index=index)
    #the 3 functions above are the core process
    #  which is actually different instances of the abstract process _struct_part_sentiment_byshared
    #the 3*2 functions below are the application of each of those
    #  for the two cases of setting the process with this subprocess and adding a subprocess
    #  it is basically the same thing but more user friendly
    #  the _struct_op variable of the _struct_part_sentiment goes into the name of the function themselves (set or add)
        
    def set_part_sentiment_byshared(self, partition_name, shared_fn, 
                                    info_name='auto', math_operation='set', index=None):          #<----------subprocess operation 
        self._struct_part_sentiment_byshared(partition_name, shared_fn, 'set',#<--- structural operation (_struct_op)
                                             info_name=info_name, math_operation=math_operation, index=index)
            
    def add_part_sentiment_byshared(self, partition_name, shared_fn, 
                                    info_name='auto', math_operation='set', index=None):          #<----------subprocess operation 
        self._struct_part_sentiment_byshared(partition_name, shared_fn, 'add',#<--- structural operation (_struct_op)
                                             info_name=info_name, math_operation=math_operation, index=index)    
        
    def set_part_sentiment_bytrend(self, partition_name, trend_type, 
                                   math_operation='set', index=None, #<----------subprocess (math) operation 
                                   **trend_kw):
        self._struct_part_sentiment_bytrend(partition_name, trend_type, 'set',#<--- structural operation (_struct_op)
                                       math_operation=math_operation, index=index, **trend_kw)           
    def add_part_sentiment_bytrend(self, partition_name, trend_type, 
                                   math_operation='set', index=None, #<----------subprocess (math) operation 
                                   **trend_kw):
        self._struct_part_sentiment_bytrend(partition_name, trend_type, 'add',#<--- structural operation (_struct_op)
                                       math_operation=math_operation, index=index, **trend_kw) 
        
    def set_part_sentiment_bymavdiff(self, partition_name, trend_type, 
                                     math_operation='set', index=None, #<----------subprocess (math) operation 
                                   **mavdiff_kw):
        self._struct_part_sentiment_bymavdiff(partition_name, trend_type, 'set',#<--- structural operation (_struct_op)
                                       math_operation=math_operation, index=index, **mavdiff_kw) 
    def add_part_sentiment_bymavdiff(self, partition_name, trend_type, 
                                     math_operation='set', index=None, #<----------subprocess (math) operation
                                   **mavdiff_kw):
        self._struct_part_sentiment_bymavdiff(partition_name, trend_type, 'add',#<--- structural operation (_struct_op)
                                       math_operation=math_operation, index=index, **mavdiff_kw)                                         

# queste due differiscono solo per set_part(con set_prt) e add_part (con add_part) 
# ed il warning se si prova a settare con add

######################################################################## shared info
    @timer_shared_computation
    def _compute_shared_info(self, **ext_kw):
        for name, fn in self.compute_shared_info.items():
            self.shared_among_agents[name]=fn(self, **ext_kw)
    
    #adding process of computing shared info among agents,
    #fn must be in signature (self, **ext_kw)
    #will be called at the beginning of each step 
    #and its return value passed to self.shared_among_agents[info_name]
    def add_compute_shared_info(self, info_name, fn):
        self.compute_shared_info[info_name]=fn
    
######################################################################## FINANCE UTILS
    def market_cap(self):
        tot_stock=0
        for ag in self.agents.values():
            tot_stock+=ag.stock
        return tot_stock*self.book.get_last_price(mode='last')
    
    def liquidity(self):
        liquidity=0
        for ag in self.agents.values():
            liquidity+=ag.wealth
        return liquidity
    
    def tot_wealth(self):
        market_cap=self.market_cap()
        liquidity=self.liquidity()
        return market_cap+liquidity
    
    #algorithm iterate backward from end to start 
    #resolution can be used only if weight!= None
    #focus='day' means the agent will refer, as the last value, to the last complete day 
#     (not last element of the array, since that array is not complete)
    #focus='transaction' means the agent will refer, as the last value, to the last complete transaction (last element of the array)
    #   this means shifting of -1 when focusing on the days

    #when calling through interval= bla, the function returns the change from last available data (end) to end-interval data
#     when calling through resolution=bla, it will evaluate the changes in bla focus
    def get_price_changes(self, start=0, end=-1, interval=None, resolution=None,  include_end='auto', focus='day', mode='absolute'):

        assert type(start)==int
        assert type(end)==int
        if end<0:
            if focus=='day':
                end=self.time+end+1
            elif focus=='transaction':
                end=len(self.price_history)+end+1
        if interval!=None:
            assert type(interval)==int
            start=end-interval  #interval is the offset from last price
            
        if not end>start:
            return []

        if focus=='transaction':
            ref_array=self.price_history
            not_include_end=0
        if focus=='day':
            ref_array=self.metric['daily_price']['close'] 
            not_include_end=1
        if include_end!='auto':
            not_include_end=int(not include_end)

        end_slice=end-not_include_end
        start_slice=max(0,start-not_include_end)
        
        if end_slice<=0:
            return []

        if resolution==None:
            if interval!=None:
                resolution=interval 
            else:
                resolution=end_slice-start_slice 
        ref_price=ref_array[end_slice:start_slice:-resolution]

        if (end_slice-start_slice)%resolution==0: #because slice never counts last element
            ref_price.append(ref_array[start_slice])    
        if debug:
            print("  start: ",start,"\t  end: ",end,"\t  start_slice:",start_slice,"\t  end_slice:",end_slice)
            print("  include_end:",include_end, "\t spec. interval:",interval,"\t  resolution:",resolution)
            print(ref_price)
        price_changes=[change_ratio(ref_price[i], ref_price[i+1], mode=mode) for i in range(len(ref_price)-1)]
        return price_changes
    
        
    def display_trend(self, deltas=[1,7,14,30,90,180, 365], focus='day', end=-1, oneperc_sent=None, 
                  weights=None, apply_norm='tanh', legend='lower right'): 

        if end==0 or end=='next_iter':
            if focus=='day':
                end=self.time
            if focus=='transaction':
                end=len(self.transaction_history)
        if focus=='day':
            if end>self.time:
                print("WARNING curr. time is", self.time)
                end=self.time
            ref_end=end-1      
        if focus=='transaction':
            if end>len(self.price_history):
                print("WARNING current number of transaction is:", len(self.price_history))
                end=len(self.price_history)
            ref_end=end
            
        if oneperc_sent is None:
            oneperc_sent=def_oneperc_sent

        plt.figure(figsize=(12,6))
        if focus=='day':
            self.plot_price(what='close',resolution=1)
            data=self.metric['daily_price']['close']
        elif focus=='transaction':
            data=self.price_history
            self.plot_price(what='transaction',resolution=1)

        if ref_end<0:
            ref_time=self.time+ref_end
        else:
            ref_time=ref_end
        colors=['b','r','c','m','y','g']*10
        ref_price=data[ref_time]
        plt.scatter(ref_time , ref_price, marker='x')#label='reference price: day '+str(ref_time))
        plt.axvline(x=ref_time+1, c='k', linestyle= 'dashdot', linewidth=1,alpha=0.1, label='agent at day'+str(ref_time+1))
        plt.plot(np.ones(len(data))*ref_price, label='reference price: day '+str(ref_time), c='k')
        
        value_array=[]
        for col, delta in zip(colors,deltas):
            comp_time=ref_time -delta
            if comp_time<0: comp_time=0
            comp_price=data[comp_time]
            value=change_ratio(ref_price, comp_price)
    #         plt.scatter(self.time+ref_end -delta , comp_price,
    #                     s=100, alpha=1, marker='x')
            plt.plot([ref_time, comp_time], [ref_price, comp_price], alpha=0.2, c=col)
                     #label="time diff:"+str(delta)+"; change ratio:{:.2f}".format(value))
            plt.plot(np.ones(len(data))*comp_price, '--', c=col, linewidth=1,
                     label="time diff "+str(delta)+"; change perc. (abs):{:.2f}".format(value))
            plt.axvline(x=comp_time, c=col, linestyle= '--', linewidth=1)
            value_array.append(value)

        if legend!='off':
            plt.legend(loc=legend)
        plt.show()

        if weights==None:
            weights=np.ones(len(deltas))
        norm_fn= _kw_to_norm(apply_norm)

        comp_array=trend_multi_basic(self, multi_distance=deltas, end=ref_end+1, oneperc_sent=oneperc_sent, 
                                     focus=focus,  return_mode='arr', apply_norm=None)
        #no normalization to get comp_array, because normalization is applied after weighted sum
        sent_trend=trend_multi_basic(self, multi_distance=deltas, end=ref_end+1, oneperc_sent=oneperc_sent, 
                                     focus=focus,  return_mode='comp', apply_norm=apply_norm, weights=weights)
        
        print("Agent at day ", ref_end+1, "reference price at day", ref_end)
        print()
        print(f"with change2sentiment scale: 1% change---> {oneperc_sent:5.3f} sentiment; normalization", apply_norm)
        print()
        tst_s=0
        for perc, val, delta, w in zip(value_array, comp_array, deltas, weights):
            print("  delta{:4}: perc. diff.:{:>8.3f};".format(delta, perc)  
                  +"  associated_sentiment{:>9.5f} with weight {:<.5f} value is {:>9.5f}".format(val, w/delta, val*w/delta)
            )
            tst_s+=val*w/delta
        print()
        print(f"trend (multi basic) sentiment for this chart: {sent_trend:10.5}") 
        print("checking with manual computation:", norm_fn(tst_s))
        
        
        
    def display_mavdiff(self, windows=[7,30,365], focus='day', end=-1, oneperc_sent=None, 
                  weights=None, apply_norm='tanh', legend='lower right'): 

        if end==0 or end=='next_iter':     
            if focus=='day':
                end=self.time
            if focus=='transaction':
                end=len(self.transaction_history)
        if focus=='day':
            if end>self.time:
                print("WARNING curr. time is", self.time)
                end=self.time
            ref_end=end-1      
        if focus=='transaction':
            if end>len(self.price_history):
                print("WARNING current number of transaction is:", len(self.price_history))
                end=len(self.price_history)
            ref_end=end
            
        if oneperc_sent is None:
            oneperc_sent=def_oneperc_sent_mavdiff

        plt.figure(figsize=(12,6))
        if focus=='day':
            self.plot_price(what='avg',resolution=1)
            data=self.metric['daily_price']['avg']
        elif focus=='transaction':
            data=self.price_history
            self.plot_price(what='transaction',resolution=1)

        if ref_end<0:
            ref_time=self.time+ref_end
        else:
            ref_time=ref_end
            
        colors=['y','m','r','b','c','k','g']*5
        ref_price=data[ref_time]
        plt.scatter(ref_time , ref_price, marker='x')#label='reference price: day '+str(ref_time))
        plt.axvline(x=ref_time+1, c='k', linestyle= 'dashdot', linewidth=1,alpha=0.1, 
                    label='agent at day '+str(ref_time+1))
        plt.plot(np.ones(len(data))*ref_price, label='reference price: day '+str(ref_time), c='k')
        plt.axvline(x=ref_time, c='k', linewidth=1, alpha=0.8)
        
        value_array=[]
        for col, window in zip(colors,windows):
            comp_time=ref_time -window
            if comp_time<0: comp_time=0
            mav_value=MAV_day(data, window=window, day=ref_time)
            diff_value=MAV_diff(data, window=window, day=ref_time)
            plt.plot(MAV(data, window=window), label="MAV"+str(window), c=col, linewidth=1)

            plt.plot(np.ones(len(data))*mav_value, '--', c=col, linewidth=1,
                     label=" window "+str(min(ref_time+1,window))+"; diff. (abs):{:4.2f}".format(diff_value))
#             plt.axvline(x=comp_time, c=col, linestyle= '--', linewidth=1)
            value_array.append(diff_value)

        if legend!='off':
            plt.legend(loc=legend)
        plt.show()

        data_size=ref_time+1
                                
        comp_array=mavdiff_multi_basic(self, multi_window=windows, end=ref_end+1, oneperc_sent=oneperc_sent, 
                                     focus=focus,  return_mode='arr', apply_norm=None)
        #no normalization is applyed for comp_array, since normalization is applied after weighted sum
        comp_sent =mavdiff_multi_basic(self, multi_window=windows, end=ref_end+1, oneperc_sent=oneperc_sent, 
                                     focus=focus,  return_mode='comp', apply_norm=apply_norm, weights=weights)
        
        norm_fn=_kw_to_norm(apply_norm)
        
        if weights is None: #auto weights are (1/window size)
            weights=np.ones(len(windows))/[min(data_size, wind) for wind in windows][:len(windows)]
        else:
            weights=np.array(weights)[:len(windows)]/[min(data_size, wind) for wind in windows][:len(windows)]
            
        print("Agent at day ", ref_end+1, "reference price at day", ref_end)
        print()
        print(f"with mavdiff2sentiment scale: 1% diff---> {-oneperc_sent:6.3f} sentiment; normalization", apply_norm)
        print()
        tst_s=0
        for perc, val, window, weight in zip(value_array, comp_array, windows, weights):
            print("  window{:4}: perc. diff.:{:>8.3f};".format(window, perc)  
                  +"  associated_sentiment{:>9.5f} with weight {:<.5f} value is {:>9.5f}".format(val, weight, val*weight)
            )
            tst_s+=val*weight
        print()
        print(f"mavdiff (multi_basic) sentiment for this chart: {comp_sent:10.5}") 
        print("checking with manual computation:", norm_fn(tst_s))
        

    
######################################################################## METRICS FUNCTION

    def get_metric(self, metricname, subname=None, partition=None, start=0, end='all'):
        if end=='all':
            end=self.time
        end+=1  #(it will include the end)
        
        if metricname in self.metric['daily_price']:
            subname=metricname
            metricname='daily_price'
            
        
        
        if partition==None:
            to_return=self.metric[metricname]
        else:
            to_return=self.partition_metric[metricname]
        
        if subname==None:
            to_return=to_return
        else:
            to_return=to_return[subname]
        
        if partition==None:
            return to_return[start:end]
        else:
            return to_return[partition][start:end]
        
        
    def add_metric(self, name, fn, fn_init=None, fn_step_init=None, fn_inside=None, add_infograph=False, **init_kw):
    
        if fn_init==None:
            fn_init=lambda mkt, **init_kw:[]
            # default initialization of metric just assign an empty array
        if fn_step_init==None:
            fn_step_init=lambda mkt, mtr, **kw : mtr.append(0)
            # default initializer of step, just appends 0 at every step
            # step initializer function, if redefined must be stated in signature def ...(self, metric_obj, **kwargs):
            # every value it needs must be passed through kwargs
        if fn_inside==None:
            fn_inside=lambda mkt, mtr, **kw: None
            # default does nothing inside step
            
        self.metric[name]=fn_init(self, **init_kw) #la inizializza direttamente, ma potrei generalizzare
        self.metric_start_step[name]=fn_step_init
#         this will be called at the start of the step
        self.metric_inside_step[name]=fn_inside
#         this will be called inside the step
        self.metric_post_step[name]=fn
#         this will be called at the end of each step
            # fn must be a function in signature def ..(self, **kwargs)
            # it will take market instance calling it as "self" and external arguments from **kwargs
            # will get its name as my_name=kwargs['metric_name']
            # for operating needs to modify value of the metric requested, acting like inside the class function herself
            # like self.metric[my_name]= ... values
            # like self.metric[my_name]=self.book.volume
        if add_infograph:
            self.add_infograph(name)
            
    def add_partition_metric(self, name, fn, fn_init=None, fn_step_init=None, fn_inside=None, add_infograph=False, **init_kw):
    
        if fn_init==None:
            fn_init=lambda mkt, **init_kw:[]
            # default initialization of metric just assign an empty array
        if fn_step_init==None:
            fn_step_init=lambda mkt, mtr, ags, **kw : mtr.append(0)
            # default initializer of step, just appends 0 at every step
            # step initializer function, if redefined must be stated in signature def ...(self, metric_obj, **kwargs):
            # every value it needs must be passed through kwargs
        if fn_inside==None:
            fn_inside=lambda mkt, mtr, ags, **kw: None
            # default does nothing inside step
        self.partition_metric[name]={}
        for part_name in self.agent_partition:
            self.partition_metric[name][part_name]=fn_init(self, **init_kw) 
        self.partition_metric_start_step[name]=fn_step_init
#         this will be called at the start of the step
        self.partition_metric_inside_step[name]=fn_inside
#         this will be called inside the step
        self.partition_metric_post_step[name]=fn
#         this will be called at the end of each step
            # fn must be a function in signature def ..(self, **kwargs)
            # it will take market instance calling it as "self" and external arguments from **kwargs
            # will get its name as my_name=kwargs['metric_name']
            # for operating needs to modify value of the metric requested, acting like inside the class function herself
            # like self.metric[my_name]= ... values
            # like self.metric[my_name]=self.book.volume
        if add_infograph:
            self.add_infograph(name)
    
    @timer_market_metric
    def ev_metric_start_step(self, **kwargs):
        #--------------------------------------------------- default global metrics
        self.metric['daily_price']['open'].append(self.book.get_last_price(mode='last'))
        self.metric['daily_price']['avg'].append(self.book.get_last_price(mode='last'))
        self.metric['daily_price']['max'].append(self.book.get_last_price(mode='last'))
        self.metric['daily_price']['min'].append(self.book.get_last_price(mode='last'))
        
        self.metric['daily_count']['buy'].append(0)
        self.metric['daily_count']['sell'].append(0)
        self.metric['daily_count']['buy_mkt'].append(0)
        self.metric['daily_count']['sell_mkt'].append(0)
        self.metric['daily_count']['nothing'].append(0)

        self.metric['daily_volume'].append(0)
        self.metric['daily_volume_money'].append(0)
        self.metric['daily_bookdepth'].append(0)
        
        #--------------------------------------------------- default partition metrics
        for partition_name in self.agent_partition:
            
            self.partition_metric['avg_sent'][partition_name].append([])
            self.partition_metric['avg_pred'][partition_name].append(0)
            self.partition_metric['daily_count']['buy'][partition_name].append(0)
            self.partition_metric['daily_count']['sell'][partition_name].append(0)
            self.partition_metric['daily_count']['buy_mkt'][partition_name].append(0)
            self.partition_metric['daily_count']['sell_mkt'][partition_name].append(0)
            self.partition_metric['daily_count']['nothing'][partition_name].append(0)
        
        #------------------------------------------------custom metrics
        for name in self.metric:
            if name not in self._default_metric:  #DEFAULT METRICS ARE EVALUATED EXPLICITALLY
                self.metric_start_step[name](self, self.metric[name], **kwargs, metric_name=name)
                
        for metric_name in self.partition_metric:
            if metric_name not in self._default_partition_metric:
                for part_name in self.agent_partition:
                    self.partition_metric_start_step[metric_name](self, self.partition_metric[metric_name][part_name],
                                                                  self.get_part_agents(part_name),
                                                                  **kwargs, metric_name=metric_name, partition_name=part_name)
    
    @timer_market_metric
    def ev_metric_inside_step(self, **kwargs):

        for name in self.metric:
            if name not in self._default_metric:  #DEFAULT METRICS ARE EVALUATED EXPLICITALLY
                self.metric_inside_step[name](self, self.metric[name], **kwargs, metric_name=name)
        
        for metric_name in self.partition_metric:
            if metric_name not in self._default_partition_metric:
                for part_name in self.agent_partition:
                    self.partition_metric_inside_step[metric_name](self, self.partition_metric[metric_name][part_name], 
                                                                   self.get_part_agents(part_name),
                                                                   **kwargs, metric_name=metric_name, partition_name=part_name)
                
    @timer_market_metric
    def ev_metric_post_step(self, **kwargs):
        #------------------------------------------------------- default global metrics
        self.metric['daily_volume'][-1]=self.book.volume
        self.metric['daily_volume_money'][-1]=self.book.volume_money
        self.metric['daily_bookdepth'][-1]=self.book.get_book_depth()

        max_pr=self.book.get_last_price(mode='last')
        min_pr=self.book.get_last_price(mode='open')
        for trs in self.book.transaction_today:
            t_price=trs['price']
            self.price_history.append(t_price)
            t_quantity=trs['quantity']
            if t_price>max_pr:  #max= ..transaction_today[:]['volume'].max() .. ?
                max_pr=t_price
            if t_price<min_pr:
                min_pr=t_price
#             print(t_price,"*",t_quantity)
            self.metric['daily_price']['avg'][-1]+=t_price*t_quantity
        self.metric['daily_price']['max'][-1]=max_pr
        self.metric['daily_price']['min'][-1]=min_pr
        self.metric['daily_price']['avg'][-1]/=max(self.metric['daily_volume'][-1],1)    
        self.metric['daily_price']['close'].append(self.book.get_last_price(mode='last'))

        for agent in self.agents.values():
            self.metric['daily_count'][agent.state['decision']][-1]+=1
            
         #------------------------------------------------------- default partition metrics       
        for agent in self.agents.values():
            self.partition_metric['daily_count'][agent.state['decision']][agent.group][-1]+=1
        for partition_name in self.agent_partition:
            part_size=len(self.agent_partition[partition_name])
            self.partition_metric['avg_sent'][partition_name][-1]=np.array(self.partition_metric['avg_sent'][partition_name][-1]).mean()
            self.partition_metric['avg_pred'][partition_name][-1]/=part_size      
            for decision in self.partition_metric['daily_count']:
                self.partition_metric['daily_count'][decision][partition_name][-1]/=part_size


        #------------------------------------------------custom metrics
        for name in self.metric:
            if name not in self._default_metric:  #DEFAULT METRICS ARE EVALUATED EXPLICITALLY
                self.metric_post_step[name](self, self.metric[name], **kwargs, metric_name=name)
        
        for metric_name in self.partition_metric:
            if metric_name not in self._default_partition_metric:
                for part_name in self.agent_partition:
                    self.partition_metric_post_step[metric_name](self, self.partition_metric[metric_name][part_name], 
                                                                 self.get_part_agents(part_name),
                                                                 **kwargs, metric_name=metric_name, partition_name=part_name)
 
                
 ####################################################################### PLOTTING FUNCTIONS 
    def _get_days_interval(self, start, end, resolution=1):
        days=list()
        for time in range(start-1+resolution, end, resolution):
            days.append(self.start_date+timedelta(time))
        return days
        
    def get_price_dataframe(self, start=0, end=-1, interval=None, resolution=1, idx_type='int'):
        assert type(start)==int
        assert type(end)==int
        if end<0:
            end=self.time+end+1
        if interval!=None:
            assert type(interval)==int
            end=start+interval
        assert end>start
    
        if idx_type=='date':
            days=self._get_days_interval(start, end, resolution)
            days=pd.DatetimeIndex(days)
            price_df=pd.DataFrame(index=days)
        else:
            price_df=pd.DataFrame(index=list(range(start-1+resolution, end, resolution)))
        
        col_val={'Open':[], 'Low':[], 'High':[], 'Close':[]}
        for day in range(start, end-resolution+1, resolution):
            t_s=day
            t_f=day+(resolution-1)

            col_val['Open'].append(self.metric['daily_price']['open'][t_s])
            col_val['Close'].append(self.metric['daily_price']['close'][t_f])
            col_val['Low'].append(min(self.metric['daily_price']['min'][t_s:t_f+1]))
            col_val['High'].append(max(self.metric['daily_price']['max'][t_s:t_f+1]))
        
        for col in col_val:  
            price_df[col]=col_val[col]

        return price_df
        
        
    def plot_price(self, what='all', show=False, start=0, end=-1, interval=None, resolution=7):
        if what=='all':
            prices=self.get_price_dataframe(start, end, interval, resolution)
#            mpl.plot(p_df, type='candle', style='yahoo')

            #define width of candlestick elements
            width = 5.0
            width2 = 1.0

            #define up and down prices
            up = prices[prices.Close>=prices.Open]
            down = prices[prices.Close<prices.Open]

            #define colors to use
            col1 = 'green'
            col2 = 'red'

            #plot up prices
            plt.bar(up.index,up.Close-up.Open,width,bottom=up.Open,color=col1)
            plt.bar(up.index,up.High-up.Close,width2,bottom=up.Close,color=col1)
            plt.bar(up.index,up.Low-up.Open,width2,bottom=up.Open,color=col1)

            #plot down prices
            plt.bar(down.index,down.Close-down.Open,width,bottom=down.Open,color=col2)
            plt.bar(down.index,down.High-down.Open,width2,bottom=down.Open,color=col2)
            plt.bar(down.index,down.Low-down.Close,width2,bottom=down.Close,color=col2)

            #rotate x-axis tick labels
#             plt.xticks(rotation=180+45, ha='right')
            plt.title("prices candels (resolution: {:.0f} days)".format(resolution))

        else:
            if what=='transaction':
                plt.plot(self.price_history)
                plt.title('transaction prices')
            else:    
                plt.plot(self.metric['daily_price'][what])
                plt.title(what+' daily price')
        plt.grid()

        if show:
            plt.show()
            
    def plot_count(self, what='all', show=False, smoothing=None, smoothing_kw={}):
        smooth_fn=smoothing_fn(smoothing, **smoothing_kw)
        sm_str=''
        if smoothing!=None:
            sm_str=' ('+smoothing+')'
        if what=='all':
            for namespec in self.metric['daily_count']:
                self.plot_count(namespec)
        else:
            plt.plot(smooth_fn(self.metric['daily_count'][what]), label=what)
            plt.title('daily count'+sm_str)
            plt.legend()
            plt.grid()
            if show:
                plt.show()
    
    def plot_partition_count(self, what,  partitions='all', show=False, smoothing='mav', smoothing_kw={'window':7}):
        smooth_fn=smoothing_fn(smoothing, **smoothing_kw)
        sm_str=''
        if smoothing!=None:
            sm_str=' ('+smoothing+')'
        N_partition=len(self.agent_partition)
        for part_name in self.agent_partition:
            if partitions=='all' or part_name in partitions:
                plt.plot(smooth_fn(self.partition_metric['daily_count'][what][part_name]), label=part_name,
                         alpha=1/N_partition+min(N_partition-1,0.2))
        plt.title("decision "+what+" partition proportion"+sm_str)
        plt.grid()
        plt.legend()
        if show:
            plt.show()
         
    def plot_metric(self, metric_name, show=False, smoothing=None, smoothing_kw={}):
        smooth_fn=smoothing_fn(smoothing, **smoothing_kw)
        sm_str=''
        if smoothing!=None:
            sm_str=' ('+smoothing+')'
        if metric_name not in self.metric:
            self.plot_partition_metric(metric_name, show=show)
        plt.plot(smooth_fn(self.metric[metric_name]), c='k')
        plt.title(metric_name+sm_str)
        plt.grid()
        if show:
            plt.show()
        
    def plot_partition_metric(self, metric, partitions='all', show=False, smoothing='mav', smoothing_kw={'window':7}):
        smooth_fn=smoothing_fn(smoothing, **smoothing_kw)
        sm_str=''
        if smoothing!=None:
            sm_str=' ('+smoothing+')'
        N_partition=len(self.agent_partition)
        for part_name in self.agent_partition:
            if partitions=='all' or part_name in partitions:
                plt.plot(smooth_fn(self.partition_metric[metric][part_name]), label=part_name, 
                         alpha=1/N_partition+min(N_partition-1,0.2))
        plt.title(metric+sm_str)
        plt.grid()
        plt.legend()
        if show:
            plt.show()
            
    #this function shows the distribution of {what} among all agent  
    #what needs to be a string recognizable by _kw_to_statefn
    #  representing some state quantity of the agents 
    #for partitionwise distribution see market.plot_part_state_dist 
    #for both partitionwise and marketwise distribution see market.plot_state_dist
    def plot_allagent_state_dist(self, what, show_avg=True):    
        state_fn=_kw_to_statefn(what)

        plt.title(f"all agent: {what} distribution")
        avg_state=0
        states=[state_fn(ag) for ag in self.agents.values()]
        hist_states =np.histogram(states,  bins=np.linspace(min(states),max(states),200))[0]
        plt.plot(np.linspace(min(states),max(states),len(hist_states)), hist_states,
                 c='k', alpha=1.)#, label="P(Sigma) at time -{:}".format(i))
        if show_avg:
            avg_states=np.array(states).mean()
            plt.axvline(x=avg_states, c='k', label='avg '+what+' {:.2f}'.format(avg_states) )
            plt.legend()
        plt.grid()

    #this function shows the distribution of {what} among all agent in {part_name} partitions
    #if part_name='all' (by default) all partitions are shown,
    #  otherwise a list of partitions name must be provided
    #what needs to be a string recognizable by _kw_to_statefn
    #  representing some state quantity of the agents 
    def plot_part_state_dist(self, what, part_name='all', show_avg=True):    
        state_fn=_kw_to_statefn(what)

        allvalues=np.array([state_fn(ag) for ag in self.agents.values()])
        maxv=allvalues.max()  #scanning just to find all values max and min 
        minv=allvalues.min()  #inefficient but does the job

        plt.title(f" partitions: {what} distribution")
        colors=['c', 'm', 'r', 'g', 'b']*5
        for col,part in zip(colors,self.agent_partition):
            if part_name!= 'all' and part not in part_name:
                continue
                print(part)
            avg_state=0
            states=[state_fn(ag) for ag in self.get_part_agents(part)]
            hist_states =np.histogram(states,  bins=np.linspace(minv ,maxv,200))[0]
            plt.plot(np.linspace(minv,maxv,len(hist_states)), hist_states, 
                     c=col, alpha=0.7)#, label="P(Sigma) at time -{:}".format(i))
            if show_avg:
                avg_states=np.array(states).mean()
                plt.axvline(x=avg_states, c=col, label=f'partition {part}: avg '+what+' {:.2f}'.format(avg_states) )
        plt.legend()
        plt.grid()
    #     plt.show()
    
    #what needs to be a string recognizable by _kw_to_statefn
    #representing some state quantity of the agents 
    #it show that state distribution on either all market and partition decomposition
    def plot_state_dist(self, what, figsize=(18,6), show_avg=True):
        plt.figure(figsize=figsize)

        plt.subplot(1,2,1)
        self.plot_allagent_state_dist(what,show_avg=show_avg)

        plt.subplot(1,2,2)
        self.plot_part_state_dist(what,show_avg=show_avg)

        plt.show()
    
    def add_infograph(self, metric_name):
        if metric_name not in self.custom_infograph:  #no check if it is in metric or partition_metric
            self.custom_infograph.append(metric_name)
    
    def show_infograph(self, price_resolution=7, price_what='all'):
        n_more_graph=len(self.custom_infograph)
        n_col=4
        n_raw=int(4+np.ceil(n_more_graph/n_col))
        plt.figure(figsize=(15,int(7+(n_raw-2)*4)))
        
        plt.subplot(n_raw,n_col,(1,6))
        self.plot_price(resolution=price_resolution, what=price_what)
        plt.subplot(n_raw,n_col,3)
        self.plot_metric('daily_volume_money')
        plt.subplot(n_raw,n_col,4)
        self.plot_metric('daily_bookdepth')
        plt.subplot(n_raw,n_col,7)
        self.plot_partition_metric('avg_sent')
        plt.subplot(n_raw,n_col,8)
        self.plot_partition_metric('avg_pred')
        plt.subplot(n_raw,n_col,(11,16))
        self.plot_count()
        plt.subplot(n_raw,n_col,9)
        self.plot_partition_count('buy')
        plt.subplot(n_raw,n_col,10)
        self.plot_partition_count('sell')
        plt.subplot(n_raw,n_col,13)
        self.plot_partition_count('buy_mkt')
        plt.subplot(n_raw,n_col,14)        
        self.plot_partition_count('sell_mkt')
        
        for k, cust_ig in enumerate(self.custom_infograph):
            plt.subplot(n_raw,n_col,17+k)
            if cust_ig in self.metric:
                self.plot_metric(cust_ig)   #la cerca nelle "metric" generali
            else:                          #se non la trova la cerca nelle metriche di partizione
                self.plot_partition_metric(cust_ig)  #se non la trova allora dà errore
                                           #LE METRICHE GENERALI E DI PARTIZIONE DEVONO AVERE NOMI DIVERSI
        plt.show()
        
  ####################################################################### OTHER FUNCTIONS         
    def set_network(self, network):
        
        if type(network)==str:
            if network=='full':
                self.network=np.ones((self.n_agents, self.n_agents))
            if network=='empty':
#                 self.network=np.zeros((self.n_agents, self.n_agents))
                self.network=None
        else:
            self.network=network                
                
    def set_spec_fn(self, when, fn):
        if when=='pre' or when=='post':
            self.special_fn[when]=fn
        
 ####################################################################### RUN FUNCTIONS   
    @timer_market_run
    def run(self, time, verbose=1, **ext_args):
        
        reset_timers()
        
        if verbose>0:
            verb_fn=tqdm
        else:
            verb_fn=lambda x:x
            
        for t in verb_fn(range(time)):
            #################################################################default kwargs
            ext_args['market']=self
            ext_args['time']=self.time
            ext_args['date']=self.date
            
            #################################################################pre-step operations
            #---------------------------clearing today valyes
            self.book.clear()
            for ag in self.agents.values(): ag.clear()        
                
            #---------------------------metrics step-initialization
            self.ev_metric_start_step(**ext_args)
            
            #--------------------------calling pre-step market special function
            self.special_fn['pre'](self, **ext_args)
            
            #--------------------------computing shared info the agents will use
            self._compute_shared_info(**ext_args)
            
            #################################################################  calling step function (agent actions)
            self._step(**ext_args)
            
            #################################################################  post step operations
            
            self.ev_metric_post_step(**ext_args)   

            #--------------------------calling post-step market special function if any
            self.special_fn['post'](self, **ext_args)

            self.time+=1
                            
    @timer_market_step
    def _step(self, **ext_args):
        self.date+= timedelta(days=1)
        #--------------------------looping THOUGHTS over all agents
                                #  this is needed when thought (sentiment and predictions) 
                                #  are ALL and IN ORDINE processed BEFORE action are evaluated 
                                #  Case use is when agents influence each other sentiments, 
                                #  based on sentiment of last iteration
                                #  self.process_preliminar is True by default
        if self.process_preliminar:
            for agent in self.agents.values():
#                 agent=self.agents[agent_id]
#                 agent_partition_name=agent.group
                agent.process_preliminar(**ext_args)

            for agent in self.agents.values():
#                 agent=self.agents[agent_id]
#                 agent_partition_name=agent.group
                agent.process_thought(**ext_args)      
                
                
        #--------------------------sorting agents in actor array
        how_many_actors=self.n_agents #DI DEFAULT AGISCONO TUTTI
#                                       #se voglio cambiare questo, devo cambiare il loop 
#                                       #e separare il processo di elaborazione di decisione dal processo di azione
#         actors_id=np.random.choice(list(self.agents.keys()), how_many_actors, replace=False)
        actors= random.sample(list(self.agents.values()), how_many_actors)

        #--------------------------looping ACTIONS over actors as sorted array of agents
#         for actor_id in actors_id:
#             actor=self.agents[actor_id]
        for actor in actors:
            actor_partition_name=actor.group
            
            if not self.process_preliminar: #se non lo ha già calcolato prima di questo loop
                actor.process_premilinar(**ext_args)
                actor.process_thought(**ext_args)
            actor.process_action(**ext_args)

            if debug: print("-"*60)
                   
            #--------------------------inside-loop metric evaluation (partition metric and general metric)
            self.partition_metric['avg_sent'][actor_partition_name][-1].append(actor.state['sentiment']) #stesse considerazioni del how_many actors
            self.partition_metric['avg_pred'][actor_partition_name][-1]+=actor.state['prediction']
            #dovrei aggiungere qui funzione generale per valutare partition metric inside step
            
            self.ev_metric_inside_step(**ext_args)
            
    def _warmup(self, t_warmup=50):
        for _ in tqdm(range(t_warmup), desc="Warmup"):
            self.book.clear()
            for ag in self.agents.values(): ag.clear() 
                
            self._warmup_step()
            
        self.book.afterwarmup_clear()
        for agent in self.agents.values():
            agent.afterwarmup_clear()
            
    def _warmup_step(self):

        how_many_actors=self.n_agents 
        actors_id=np.random.choice(range(0, self.n_agents), how_many_actors, replace=False)

        for actor_id in actors_id:
            actor=self.agents[actor_id]
            actor.process_thought()
            actor.process_action()
            
####################################################################### EXTERNAL FUNCTIONS 


# ------------------------------------------------------- generic utility

def _kw_to_sampler(sampler_type, sample_size=1, return_type='iterable', **kw):    
    if not isinstance(sampler_type, str):
        return lambda: [sampler_type]*sample_size   
    if sampler_type.lower() in ['deltadirac', 'point', 'unique', 'same']:
        avg=kw['avg']
        return lambda: [avg]*sample_size
    if sampler_type.lower() in ['uniform', 'unif', 'uni']:
        low=kw['low']
        high=kw['high']
        return lambda: np.random.uniform(low=low, high=high, size=sample_size)
    if sampler_type.lower() in ['gauss', 'gaussian', 'normal', 'norm']:
        print("not implemented")        
    if sampler_type.lower() in ['beta']:
        print("not implemented")        
    if sampler_type.lower() in ['scale free', 'scale_free']:
        print("not implemented")        
    if sampler_type.lower() in ['pois','poiss', 'poisson', 'poissonian']:
        avg=kw['avg']
        return lambda: np.random.poisson(lam=avg, size=sample_size)

def _kw_to_norm(apply_norm, **norm_kw):
    norm_max=norm_kw.get('norm_max',+1.)
    norm_min=norm_kw.get('norm_min',-1.)
    steepness=norm_kw.get('steepness',1.)
#     shift==norm_kw.get('shift',0.)
            
    if apply_norm=='tanh':
        return lambda x: norm_max*np.tanh(x*steepness)
    elif apply_norm=='sigm':
        return lambda x: (1 / (1 + np.exp(-x)))*2-1
    elif apply_norm=='clip':
        return lambda x: np.clip(x,norm_min,norm_max)
    else:
        return lambda x:x
    
def _kw_to_statefn(what):
    if 'liq' in what:
        state_fn= lambda ag: ag.wealth
    if 'stock' in what:
        state_fn= lambda ag: ag.stock
    if 'sigm' in what:
        state_fn= lambda ag: ag.sigma()
    if 'eq' in what:
        state_fn= lambda ag: ag.equilibrium()
    if 'dist' in what and 'eq' in what:
        state_fn= lambda ag:ag.distance_from_eq()
    if 'tot' in what and 'wealth' in what:
        state_fn= lambda ag: ag.tot_wealth()
    if 'sent' in what:
        state_fn= lambda ag: ag.state['sentiment']
    if 'pred' in what:
        state_fn= lambda ag: ag.state['prediction']
        
    #raise error if no correspondence is found 
    return state_fn
            
# ------------------------------------------------------- time-series utility
    
def change_ratio(a,b,mode='absolute'): #(a=last, b=second_last)
    if 'abs' in mode: #max(last, second)/min(last, second)
        try:
            ratio=(max(a,b)/min(a,b)-1)*100
        except:
            return 0
        if a>b:
            return ratio
        else:
            return -ratio
    if 'rel' in mode: #last/second_last 
        return (a/b-1)*100
    
#sentiment associated by a 1% price increase
def_oneperc_sent=0.02

# oneperc_sent is the sentiment associated with a 1% growth
# this function set the scale from passing to obs. ratio to sentiment
# is the function : observed change (in fractio) to assignend sentiment
#num value is output value for den value
def change2sentiment(change, oneperc_sent=None):
    if oneperc_sent is None:
        oneperc_sent=def_oneperc_sent
    scale=oneperc_sent
    return change*scale
c2s=change2sentiment

#function (perc_change, day_distance)--->sentiment    CURRENTLY NOT USING
# def change_day2sentiment(change, day_dist, fn_change='def', fn_day='def', **cd2s_kw):
#     if fn_change=='def':
#         scale=cd2s_kw.get('scale', def_oneperc_sent)
#         fn_change=lambda c: c2s(c)
#     if fn_day=='def':
#         coeff=cd2s_kw.get('coeff', 1/2)
#         fn_day=lambda x, d: x/d**coeff   
#     return fn_day(fn_change(change), day_dist)
# cd2s=change_day2sentiment

    
#return a value that extimates the sentiment due to change with respect to that given past event   
#takes the percentage {mode} value of {focus (es day)} difference of {end} from {distance},
#convert it with change2sentiment function using scale {oneperc_sent}
#then apply norm function given by _kw_to_norm({apply_norm}, norm_keyword})
#if provided distance is further than first past data recorded, it will provide the results from first past record
def trend_basic(market, distance, end=-1, oneperc_sent=None,
                focus='day', mode='abs', apply_norm='tanh', **norm_kw):
    if oneperc_sent is None:
        oneperc_sent=def_oneperc_sent

    norm_fn=_kw_to_norm(apply_norm, **norm_kw)
    
    pc=market.get_price_changes(end=end, interval=distance, focus=focus, mode=mode)
    #all values are taken as percentage, meaning 1,2% instead of 0.012
    #all values are returned as percentage, meaning 1,2% instead of 0.012
    if not len(pc):
        pc=market.get_price_changes(end=end, focus=focus, mode=mode)
        #default behaviour of get_price_changes returns changes from start=0 to end
    if len(pc):
        scaled_pc=change2sentiment(pc[0], oneperc_sent=oneperc_sent)
        return norm_fn(scaled_pc)       
    return 0
#return a value (or array) that extimates the sentiment due to change with respect to that given past eventS   
#takes the percentage {mode} value of {focus (es day)} differences of {end} from MULTIple past events {multi_distance},
#for each of these value it calls the trend_basic converted to sentiment with change2sentiment function using scale {oneperc_sent}
#return it like this if return_mode='array'
#otherwise if return_mode='avg'
#with {weight} if provided, or weight_i will be 1/distance
#so that for each time distance, the algorithm looks at average growth
#it computes the weighted sum of those value
#then apply norm function given by _kw_to_norm({apply_norm}, norm_keyword})
def trend_multi_basic(market, multi_distance=[1,7,14,30,90,180, 365], end=-1, oneperc_sent=None,
                      focus='day', mode='abs', weights=None, return_mode='composite', apply_norm='tanh', **norm_kw):
    if oneperc_sent is None:
        oneperc_sent=def_oneperc_sent
    
    norm_fn=_kw_to_norm(apply_norm, **norm_kw)
        
    base_trend=[trend_basic(market, dist, end=end, oneperc_sent=oneperc_sent, #scaling perc-->sentiment is performed here
                            focus=focus, mode=mode, apply_norm=None) for dist in multi_distance]
#     base_trend=[btv for btv in base_trend if btv is not None] 
    base_trend=np.array(base_trend)
    
    N_trend=len(base_trend)
    #values returned are:
    if 'arr' in return_mode:   # -array of variational percentage
        if N_trend==0: return []
        to_return= base_trend  #after applying c2s, and normalization before returning 

    elif 'comp' in return_mode:   # -weighted average of basic trend
                                 
        if N_trend==0: return 0
        if weights is None: #auto weights are (1/distance)
            weights=np.ones(N_trend)/multi_distance[:N_trend]
        else:
            weights=weights[:N_trend]/multi_distance[:N_trend]
            
        to_return= (weights*base_trend).sum()#/len(multi_distance) #
        
    return norm_fn(to_return)

#computes average of data in a window of {window} 
def MAV(data, window=7):
    if isinstance(data, Market):
        data=market.get_metric('daily_price','avg')
#     mav=[ np.array(data[max(0,i-window+1):i+1]).mean() for i in range(len(data))]
#     faster implementation
    mav=[]
    data=np.array(data)
    for i in range(len(data)):
        if i<window:
            mav.append(data[max(0,i-window+1):i+1].mean())
        else:
            mav.append(mav[-1]+(data[i]-data[i-window])/window)
    return np.array(mav)

def smoothing_fn(sm_type, **sm_kw):
    if isinstance(sm_type, str):
        if 'mav' in sm_type.lower():
            window=sm_kw.get('window', 7)
            return lambda data: MAV(data, window)
        
    return lambda data: data

#mav of {window} at given {day}
def MAV_day(data, window=7, day=-1):   #if window is over data size, window is set to maximum possible size
    if isinstance(data, Market):
        data=market.get_metric('daily_price','avg')
    if day<0:
        day=len(data)+day
    if day<0:
        return 0
    if day>len(data):
        print(f"WARNING asking mav_diff for day {day}, provided data size {len(data)}")
        print("ref day will be set to maximum possible")
        day=len(data)-1
    if day==len(data):
        day=len(data)-1
    window_start=max(0,day-window+1)
    window_end=day+1
#     print(interval_start, interval_end, day)
    assert (window_end-window_start)<=window
    if window<=day: assert (window_end-window_start)==window
        
    mav_day=np.array(data)[window_start:window_end].mean()

    return mav_day
MAV_day(list(range(10)))

#difference from value at {day} position in provided {data}, from mav{window} in that day
def MAV_diff(data, window=7, day=-1, mode='abs'):
#     print(day)
    #getting day data
    if isinstance(data, Market):
        data=market.get_metric('daily_price','avg')
    if not len(data): #if no data is provided
        return 0
    if day<0:
        day=len(data)+day
    if day<0:
        return 0
    if day>len(data):
        print(f"WARNING asking mav_diff for day {day}, provided data size {len(data)}")
        print("ref day will be set to maximum possible")
        day=len(data)-1
    if day==len(data):
        day=len(data)-1
#     print(day)
#     print(len(data))
    day_data=data[day] 
        
    #getting mav data
    mav_data=MAV_day(data, window, day)
 
    #comparing mav data with day data
    perc_dist=change_ratio(day_data, mav_data, mode)
    
    return perc_dist

#sentiment associated by a 1% price increase
def_oneperc_sent_mavdiff=0.05

# oneperc_sent is the sentiment associated with a 1% growth
# this function set the scale from passing to obs. ratio to sentiment
# is the function : observed change (in fractio) to assignend sentiment
#num value is output value for den value
def mavdiff2sentiment(mavdiff, oneperc_sent=None):
    if oneperc_sent is None:
        oneperc_sent=def_oneperc_sent_mavdiff
    scale=oneperc_sent
    return mavdiff*scale
d2s=change2sentiment

    
def mavdiff_basic(data, window=7, end=-1, oneperc_sent=None,
                focus='day', mode='abs', apply_norm='tanh', **norm_kw):
    if oneperc_sent is None:
        oneperc_sent=def_oneperc_sent_mavdiff

    norm_fn=_kw_to_norm(apply_norm, **norm_kw)
    
    if isinstance(data, Market): 
        if focus=='day':
            data=data.get_metric('daily_price','avg')[:-1]
        if focus=='transaction':
            data=data.price_history
            
    if not len(data):
        return 0
        
    mav_diff=MAV_diff(data, window=window, day=end, mode=mode)    
    mav_diff*=-1 #sentiment will be positive if price is under mav
    
    sd=mavdiff2sentiment(mav_diff, oneperc_sent)
    return norm_fn(sd)
    
def mavdiff_multi_basic(data, multi_window=[7,30,365], end=-1, oneperc_sent=None,
                focus='day', mode='abs', weights=None, return_mode='composite',apply_norm='tanh', **norm_kw):
    if oneperc_sent is None:
        oneperc_sent=def_oneperc_sent_mavdiff

    norm_fn=_kw_to_norm(apply_norm, **norm_kw)
    
    if isinstance(data, Market): 
        if focus=='day':
            data=data.get_metric('daily_price','avg')[:end] #remove last element
        if focus=='transaction':
            data=data.price_history
        
    mav_array=[mavdiff_basic(data, window=w, end=end, oneperc_sent=oneperc_sent, #<--- conversion diff2sentiment performed here
                             focus=focus, mode=mode, apply_norm=None) for w in multi_window]    
#     print(mav_array)
    mav_array=np.array(mav_array)
    
    N_mav=len(mav_array)
    #values returned are:
    if 'arr' in return_mode:   # array of difference percentage
        if N_mav==0: return []
        to_return= mav_array  #after applying d2s, without normalization,   

    elif 'comp' in return_mode:   # weighted average of basic trend
        data_size=len(data)
                                 
        if N_mav==0: return 0
        if weights is None: #auto weights are (1/min(window_size, maximum_possible window_size)
            weights=np.ones(N_mav)/[max(1,min(data_size, wind)) for wind in multi_window][:N_mav]
        else:
            weights=np.array(weights)[:N_mav]/[max(1,min(data_size, wind)) for wind in multi_window][:N_mav]
            
        to_return= (weights*mav_array).sum()#/len(multi_distance) #
        
    return norm_fn(to_return)

###################################################################### built-in function (builder) for agent process
    
# index can be None, in which case state[state_name] is taken as a single value variable
# if index is not none, the program will try to access state[state_name][index]
# index can be int, in which case state[state_name] is taken as a list and object at index value is modified
# index can be a string, in which case state[state_name] is taken as a dictionary
    
def _ag_state_set(ag, state_name, index, v):
    if index is None:
        ag.state[state_name]=v
    else:
        ag.state[state_name][index]=v
    
def _ag_state_add(ag, state_name, index, v):
    if index is None:
        ag.state[state_name]+=v
    else:
        ag.state[state_name][index]+=v
    
def _ag_state_mult(ag, state_name, index, v):
    if index is None:
        ag.state[state_name]*=v
    else:
        ag.state[state_name][index]*=v

def _ag_state_map(ag, state_name, index, map_fn):
    if index is None:
        ag.state[state_name]=map_fn(ag.state[state_name])
    else:
        ag.state[state_name][index]=map_fn(ag.state[state_name][index])

#in cases where an operation (subprocess) is defined as abstract
#   such as the cases where the same operation can be performed on different indexes of the state dict
#   or such as the cases where the same operation can be performed with add/mult and so on
#   this function makes so that once the abstract function is instantiated,
#   the program will know what to do and which operation to execute,
#   instead of re-evaluating it at each call which can waste time
def _kw_to_agent_op(state_name, op, index): 
    """
    state name:
        is the name of the agent state you want to modify
        each agent has a dictionary of agent.state, so variable agent.state[state_name] is modified
    op:
        is the operation you want to apply to agent.state[state_name]
        can be set, add, mult[iply] or map
    index:
        is the indication on  how to access variable agent.state[state_name] information
        can be None, in which case state[state_name] is taken as a single value variable
        if index is not none, the program will try to access state[state_name][index]
           can be int, in which case state[state_name] is taken as a list and object at index value is modified
           can be a string, in which case state[state_name] is taken as a dictionary
    
    """
    if op=='set':
        return lambda ag, v: _ag_state_set(ag, state_name, index, v)
    if op=='add':
        return lambda ag, v: _ag_state_add(ag, state_name, index, v)
    if op=='mult':
        return lambda ag, v: _ag_state_mult(ag, state_name, index, v)
    if op=='map':
        return lambda ag, m: _ag_state_map(ag, state_name, index, m)

__coord_agent_op='set'

#generic function whould have signature(agent, process, operation, index, value)

#at every step, the sentiment value is set at 'sent_value'
#should be the first in the process     
def sentiment_bias(sent_value, math_operation=None, index=None):
    if math_operation is None:
        math_operation=__coord_agent_op 
    agent_op=_kw_to_agent_op('sentiment', math_operation, index) 
    def routine(agent, value, **kw):
        agent_op(agent,value)
        return value
        
    subprocess_fn= lambda a, **kw: routine(a, sent_value, **kw)
    return Operation(subprocess_fn, sub_name='bias', name='sentiment', mode='explicit',
                      math_operation=math_operation, sent_value=sent_value)

#at step 'at_time' the sentiment is set at 'sent_value'
#should be the last in the process
def sentiment_set_at_time(sent_value, at_time=0, math_operation=None, index=None):
    if math_operation is None:
        math_operation=__coord_agent_op 
    agent_op=_kw_to_agent_op('sentiment', math_operation, index)    
    def routine(agent, value, at_time, **kw):
        if kw['time']==at_time:
            agent_op(agent,value)
            return value
             
    subprocess_fn= lambda a, **kw: routine(a, sent_value, at_time, **kw)
    return Operation(subprocess_fn, sub_name='set_at_time', name='sentiment', mode='explicit',
                      math_operation=math_operation, sent_value=sent_value, at_time=at_time)

#at step 0 the sentiment is set at 'sent_value'
#should be the last in the process
def sentiment_start_at_value(sent_value, math_operation=None, index=None):
    if math_operation is None:
        math_operation=__coord_agent_op 
    agent_op=_kw_to_agent_op('sentiment', math_operation, index) 
    def routine(agent, value, **kw):
        if kw['time']==0:
            agent_op(agent,value)
            return value
        return 0.
    subprocess_fn= lambda a, **kw: routine(a, sent_value, **kw)
    return Operation(subprocess_fn, sub_name='start_at_value', name='sentiment', mode='explicit',
                      math_operation=math_operation, sent_value=sent_value)

def sentiment_add_at_time(sent_value, at_time, math_operation='add', index=None):    
    math_operation='add'                    #this function does not need a math_operation=set mode
    agent_op=_kw_to_agent_op('sentiment', math_operation, index) 
    def routine(agent, value, at_time, **kw):
        if kw['time']==at_time:
            agent_op(agent,value)
            return value
    subprocess_fn= lambda a, **kw: routine(a, sent_value, at_time, **kw)
    return Operation(subprocess_fn, sub_name='add_at_time', name='sentiment', mode='explicit',
                      math_operation=math_operation, sent_value=sent_value, at_time=at_time)

#every step, the sentiment is set to function value
#function can be array of values, or a function of time
#could be the first (if used as time-dependent bias) 
#could be the last (if used as fixed at that time)
def sentiment_exogen(function, math_operation=None, index=None):
    if math_operation is None:
        math_operation=__coord_agent_op
    agent_op=_kw_to_agent_op('sentiment', math_operation, index) 
    def routine(agent, function, **kw):
        curr_time=kw['time']
        try:
            value=function(curr_time) 
        except:
            value=function[curr_time]
        agent_op(agent,value)
        return value
    subprocess_fn= lambda a, **kw: routine(a, function, **kw)
    return Operation(subprocess_fn, sub_name='exogen', name='sentiment', mode='explicit',
                      math_operation=math_operation)

#periodic fn needs to be defined in interval [0,1[ for dominio
#the periodic fn input will be (current_time%period)/period
def sentiment_periodic(periodic_fn, strenght, period=365, math_operation=None, index=None):
    if math_operation is None:
        math_operation=__coord_agent_op 
    agent_op=_kw_to_agent_op('sentiment', math_operation, index) 
    def routine(agent, periodic_fn, strenght, period, **kw):
        time=kw['time']
        value=strenght*periodic_fn((time%period)/period)
        agent_op(agent,value) 
        return value
    subprocess_fn= lambda a, **kw: routine(a,  periodic_fn, strenght, period, **kw)
    return Operation(subprocess_fn, sub_name='periodic', name='sentiment', mode='explicit',
                      math_operation=math_operation, period=period, strenght=strenght)

#additive noise: takes previous sentiment value and add a random value
#it clips value in -1,1 in case random extraction goes over sentiment dominio
def sentiment_add_noise(noise_sigma=0.01, math_operation='mult', index=None):  
    math_operation='mult'
    agent_op=_kw_to_agent_op('sentiment', math_operation, index)
    agent_op_map= _kw_to_agent_op('sentiment', 'map', index)
    def routine(agent, noise_sigma, **kw):
        noise_add=np.random.normal(0,noise_sigma)
        agent_op(agent, 1+noise_add)
        agent_op_map(agent, lambda v: np.clip(v, -1,1))
        
        return noise_add
    subprocess_fn= lambda a, **kw: routine(a, noise_sigma, **kw)
    return Operation(subprocess_fn, sub_name='add_noise', name='sentiment', mode='explicit',
                      math_operation=math_operation, noise_sigma=noise_sigma)
    
def sentiment_decay(perc, math_operation='mult', index=None):
    math_operation='mult'
    agent_op=_kw_to_agent_op('sentiment', math_operation, index)  
    def routine(agent, perc, **kw):
        agent_op(agent, 1-perc)
        return perc
    subprocess_fn= lambda a, **kw: routine(a, perc, **kw)
    return Operation(subprocess_fn, sub_name='decay', name='sentiment', mode='explicit',
                      math_operation=math_operation, perc=perc)
    
# #mode could be static
# def sentiment_from_neighbours(mode='static', neigh_filter= lambda x:x, neigh_sentiment_op= lambda x:x.mean()):
#     def routine(agent, **kw):
#         network=kw['market'].network
#         if mode=='static':
#             agent.state['sentiment']=0
#         elif 'autoreg' not in mode:
#             print("WARNING unknown network op. mode")
# #network matrix element N[i,j] represent "how much i is unfluenced by j": info flow i<----j
#         neighbours=[neigh for 
            

#if some value is computed in market -then stored in mkt.shared_among_agents dict
#this routine can retrieve this info and set that as its sentiment value
#NEED SOME mkt.add_compute_shared_info(function, info_name) TO WORK
#since no shared info is computed by default
def sentiment_take_from_shared(info_name, math_operation=None, index=None):
    if math_operation is None:
        math_operation=__coord_agent_op
    agent_op=_kw_to_agent_op('sentiment', math_operation, index)
    def routine(agent, info_name, **kw):
        value=kw['market'].shared_among_agents[info_name]
        agent_op(agent,value)
        return value
    subprocess_fn= lambda a, **kw: routine(a, info_name, **kw)
    return Operation(subprocess_fn, sub_name='take_from_shared_computation', name='sentiment', mode='explicit',
                      math_operation=math_operation, info_name=info_name)

def sentiment_map(map_fn, index=None):
    agent_op= _kw_to_agent_op('sentiment', 'map', index)
    def routine(agent, map_fn, **kw):
        agent_op(agent, map_fn)
        return 'applied custom mapping'
    subprocess_fn= lambda a, **kw: routine(a, map_fn, **kw)
    return Operation(subprocess_fn, sub_name='mapping', name='sentiment', mode='explicit',
                      math_operation='map')      
    
def sentiment_clip(minv=-1., maxv=+1., index=None):
    agent_op= _kw_to_agent_op('sentiment', 'map', index)
    def routine(agent, minv, maxv, **kw):
        agent_op(agent, lambda v: np.clip(v, minv, maxv))
        return 'applied clipping'
    subprocess_fn= lambda a, **kw: routine(a,  minv, maxv, **kw)
    return Operation(subprocess_fn, sub_name='norm_clip', name='sentiment', mode='explicit',
                      minv=minv, maxv=maxv)

def sentiment_sigm(steepness=1., index=None):
    agent_op= _kw_to_agent_op('sentiment', 'map', index)
    def routine(agent, steepness, **kw):
#         s=agent.state['sentiment']
#         agent.state['sentiment']=(1 / (1 + np.exp(-s*steepness)))*2-1
        agent_op(agent, lambda s: (1 / (1 + np.exp(-s*steepness)))*2-1 )
        return agent.state['sentiment']
    subprocess_fn= lambda a, **kw: routine(a,  steepness, **kw)
    return Operation(subprocess_fn, sub_name='norm_sigmoid', name='sentiment', mode='explicit',
                      steepness=steepness)

def sentiment_tanh(steepness=1., index=None):
    agent_op= _kw_to_agent_op('sentiment', 'map', index)
    def routine(agent, steepness, **kw):
#         s=agent.state['sentiment']
#         agent.state['sentiment']=np.tanh(s*steepness)
        agent_op(agent, lambda s: np.tanh(s*steepness) )
        return agent.state['sentiment']
    subprocess_fn= lambda a, **kw: routine(a,  steepness, **kw)
    return Operation(subprocess_fn, sub_name='norm_tanh', name='sentiment', mode='explicit',
                      steepness=steepness)

def sentiment_normalize(mode='clip', *args, **kwargs):
    if mode=='clip':
        return sentiment_clip(*args, 'kwargs')
    if mode=='sigm':
        return sentiment_sigm(*args, 'kwargs')
    if mode=='tanh':
        return sentiment_tanh(*args, 'kwargs')

################################################################################ built-in inline metric to add
def mtr_sct_sigma(self, metric_obj, **kwargs):
    time=kwargs['market'].time
    today_sigma=[]
    for agent in self.agents.values():
        today_sigma.append(agent.sigma())     
    metric_obj[-1]=np.array(today_sigma)      
    
def mtr_avg_sigma(self, metric_obj, **kwargs):
    today_sigma=[]
    for agent in self.agents.values():
        today_sigma.append(agent.sigma())
    metric_obj[-1]=np.array(today_sigma).mean()
    
def mtr_sct_eq_dist(self, metric_obj, **kwargs):
    today_eq_dist=list()
    for agent in self.agents.values():
        today_eq_dist.append(agent.distance_from_eq())
    metric_obj[-1]=np.array(today_eq_dist)
    
def mtr_avg_eq_dist(self, metric_obj, **kwargs):
    today_eq_dist=list()
    for agent in self.agents.values():
        today_eq_dist.append(agent.distance_from_eq())
    metric_obj[-1]=np.array(today_eq_dist).mean()
    
def mtr_sct_abs_eq_dist(self, metric_obj, **kwargs):
    today_abs_eq_dist=list()
    for agent in self.agents.values():
        today_abs_eq_dist.append( np.abs(agent.distance_from_eq()) )
    metric_obj[-1]=np.array(today_abs_eq_dist)
  
def mtr_avg_abs_eq_dist(self, metric_obj, **kwargs):
    today_abs_eq_dist=list()
    for agent in self.agents.values():
        today_abs_eq_dist.append( np.abs(agent.distance_from_eq()) )
    metric_obj[-1]=np.array(today_abs_eq_dist).mean()
    
def mtr_abs_avg_eq_dist(self, metric_obj, **kwargs):
    today_eq_dist=list()
    for agent in self.agents.values():
        today_eq_dist.append(agent.distance_from_eq())
    metric_obj[-1]=np.abs(np.array(today_eq_dist).mean())
    
def mtr_closeopen_variation(self, metric_obj, mode='abs', **kwargs):
    fin_price=self.metric['daily_price']['close'][-1]
    str_price=self.metric['daily_price']['open'][-1]
    price_var=change_ratio(fin_price, str_price, mode=mode)
    metric_obj[-1]=price_var
    
def mtr_market_cap(self, metric_obj, **kwargs):
    market_cap=self.market_cap()
    metric_obj[-1]=market_cap
    
def mtr_market_liquidity(self, metric_obj, **kwargs):
    market_liquidity=self.liquidity()
    metric_obj[-1]=market_liquidity

def mtr_market_totwealth(self, metric_obj, **kwargs):
    market_totwealth=self.tot_wealth()
    metric_obj[-1]=market_totwealth   
    
def mtr_part_sigma_average(self, metric_obj, part_agents, **kw):
    avg_sigma=0
    for agent in part_agents:
        avg_sigma+=agent.sigma()
    metric_obj[-1]=avg_sigma/len(part_agents)
    
def mtr_part_liquidity(mkt, mtr_obj, part_agents, **kw):
    part_liquid=0
    for agent in part_agents:
        part_liquid+=agent.wealth
    mtr_obj[-1]=part_liquid

def mtr_part_totwealth(mkt, mtr_obj, part_agents, **kw):
    part_wealth=0
    for agent in part_agents:
        part_wealth+=agent.wealth+agent.stock*mkt.book.get_last_price(mode='last')
    mtr_obj[-1]=part_wealth
    
def mtr_part_stockflow(mkt, metric_obj, ag, **kw):
    this_part=kw['partition_name']
    transaction_data=mkt.book.transaction_today
    for tr_data in transaction_data:
        b_part=mkt.agents[tr_data['b_id']].group
        s_part=mkt.agents[tr_data['s_id']].group
        if b_part==this_part:
            if s_part==this_part:
                continue
            metric_obj[-1]+=tr_data['price']*tr_data['quantity']  #buying stock means stock entering partition 
        if s_part==this_part:
            metric_obj[-1]-=tr_data['price']*tr_data['quantity'] #selling stock means stock leaving partition 

# ------------------------------------------------------- built-in plot routine for predefined metrics
def plot_sigma_scatter(mkt, show_avg=False, ext_sigma=None):
    
    sigma_scatter=[]
    sigma_avg=[]
    for day, day_val in enumerate(mkt.get_metric('sigma_scatter')):
        sigma_scatter+=[[day, t_day_val] for t_day_val in day_val]
        sigma_avg.append(day_val.mean())
        
    plt.title("Sigma scatter plot")
    plt.scatter([v[0] for v in sigma_scatter], [v[1] for v in sigma_scatter], s=0.0002, alpha=0.9, color='k')
    if show_avg:
        plt.plot(sigma_avg, color='r', label='avg Sigma')
    if ext_sigma!=None:
        plt.plot(np.ones(len(sigma_avg))*ext_sigma, 'r--', alpha=0.8, label='ext avg Sigma')
    if show_avg or ext_sigma!=None:
        plt.legend(loc='lower left')
    plt.grid()

def plot_sigma_distribution(mkt, show_avg=False, ext_sigma=None, time_sample=10):
    plt.title("Sigma distribution in latest iterations")
    avg_sigma=0
    for i in range(1,time_sample+1):
        time_sigma=mkt.get_metric('sigma_scatter')[-i]
        hist_time_sigma =np.histogram(time_sigma,  bins=np.linspace(0.,1.,100))[0]
        plt.plot(np.linspace(0,1,len(hist_time_sigma)), hist_time_sigma/1000, alpha=0.7)#, label="P(Sigma) at time -{:}".format(i))
        if show_avg:
            avg_sigma+=np.array(time_sigma).mean()
    if show_avg:
        avg_sigma/=(time_sample)
        plt.plot(avg_sigma*np.ones(100), np.linspace(0,max(hist_time_sigma/1000),100), 'k--',
                 label='avg (of avg) Sigma:{:.5f}'.format(avg_sigma) )
    if ext_sigma!=None:
        plt.plot(ext_sigma*np.ones(100), np.linspace(0,max(hist_time_sigma/1000),100), 'r--',
                 label='ext avg Sigma: {:.5f}'.format(ext_sigma) )
    if show_avg or ext_sigma!=None:
        plt.legend(loc='lower left')
    plt.grid()
    

############################################################################## DIAGNOSTIC

def test_agent_decision(Sigma, Theta, to_return='sigma', d_kw=None):
    if d_kw==None:
        d_kw=default_parameter['decision']
    ag=Qecon()
    ag.decision_kw=d_kw
    puppet_price=100
    bk=Book(start_price=puppet_price)        
    ag.book=bk

    #setting his sigma
    tot_wealth=1000   #s*price/1000=Sigma
    ag.stock=Sigma*10
    ag.wealth=tot_wealth-ag.stock*100
    
    old_stock=ag.stock
    old_wealth=ag.wealth
    
    #setting his theta
    ag.state['sentiment']=Theta
    for subpr in ag.process['prediction']:
        subpr(ag)
    for subpr in ag.process['decision']:
        subpr(ag)    
    
#     print(ag.state['decision'])
    if ag.state['decision']=='buy':
        ag.process_buy(ag.price, ag.quantity)
    if ag.state['decision']=='sell':
        ag.process_sale(ag.price, ag.quantity)
    
    if to_return=='sigma':
        return ag.sigma()
    elif 'quantity' in to_return:
        if 'buy' in ag.state['decision']:
            if 'scaled' in to_return:
                return ag.quantity/(old_wealth/100)
            return ag.quantity
        if 'sell' in ag.state['decision']:
            if 'scaled' in to_return:
                return -ag.quantity/old_stock
            return -ag.quantity
        if 'nothing' in ag.state['decision']:
            return 0

def test_agent_theta(Theta, sample_size=100, to_return='sigma', d_kw=None):
    if d_kw==None:
        d_kw=default_parameter['decision']
    sample_results={}
    for sigma_old in np.linspace(0.001,0.999,101):
        sample_results[sigma_old]=[]
        for i in range(sample_size):
            sample_results[sigma_old].append(test_agent_decision(sigma_old, Theta, to_return=to_return, d_kw=d_kw))
    return sample_results

observer={}

def _plot_sigma_test(sigma_new_theta, sample_size, display='sigma', theta='unk'):
    for sigma_old in sigma_new_theta:
        plt.scatter(np.ones(sample_size)*sigma_old, sigma_new_theta[sigma_old], c='k', marker='x', s=10/sample_size)

        pos_avg=np.array([val for val in sigma_new_theta[sigma_old] if val>0]).mean()
        neg_avg=np.array([val for val in sigma_new_theta[sigma_old] if val<0]).mean()

        observer[theta].append([sigma_old,(-neg_avg-pos_avg)/(-neg_avg+pos_avg)])

    if display=='sigma':
        plt.ylim([0,1])
    if display=='quantity':
        plt.ylim([-10,10])
    plt.xlim([0,1]) 
    
#run a diagnostic of agent decisions outcome  (display can be quantity or sigma)
def diagnostic_agent_decision(sample_size=100, display='quantity', d_kw=None):
    if d_kw is None:
        d_kw=default_parameter['decision']
        
    print("testing agent decision:")
    print("decision parameters:")
    for parname in d_kw:
        print("   ",parname,":", d_kw[parname])
    
    plt.figure(figsize=(16,16))
    count=0
    for theta in tqdm(np.linspace(-1,1,11)):
        count+=1
        plt.subplot(3,4,count)
        plt.title("Traded quantity $q$ when $P={:.2f}$".format((theta+1)/2))
        plt.xlabel("$\Sigma$")
        if count%4==4:
            plt.ylabel("$q$", rotate=90)
        observer[theta]=[]
        
        sigma_new_theta=test_agent_theta(theta, sample_size, to_return=display)
        _plot_sigma_test(sigma_new_theta, sample_size, display=display, theta=theta)
        
    plt.show()
    
    #NB observer{theta:[sigma (0,1), avgfractio (-1,1)]
    if 'quantity' in display:
        plt.figure(figsize=(16,16))
        avg_balance={'theta':[], 'avg_dist':[], 'avg_dist_in_eq_range':[]}
        for k, theta in enumerate(observer):
            plt.subplot(3,4,k+1)
            plt.xlabel("$\Sigma_t$")
            prediction=(theta+1)/2
            plt.title("prediction $P_t$={:.2f}".format(prediction))
            plt.scatter([v[0] for v in observer[theta]],[v[1] for v in observer[theta]], s=5)
            avg_balance['theta'].append(prediction)
            avg_balance['avg_dist'].append(np.array([v[1]+(v[1]-(v[0]*2-1)) for v in observer[theta]]).mean())
            plt.scatter([v[0] for v in observer[theta]],[v[1] for v in observer[theta]], s=5)
                      #plotting difference from expected (enfatized)
            plt.plot([v[0] for v in observer[theta]],[v[1]+(v[1]-(v[0]*2-1))*10 for v in observer[theta]], c='r')
            plt.axvline((theta+1)/2, color='k', linestyle='--')
            avg_dist_in_eq_range=np.array([v[1]-(v[0]*2-1) for v in observer[theta] 
                                          if np.abs(v[0]-prediction)<=0.1]).mean()
            avg_balance['avg_dist_in_eq_range'].append(avg_dist_in_eq_range)
        #     plt.plot(avg_balance['theta'],avg_balance['dist_from_eq'], c='k')

        plt.subplot(3,4,12)
        plt.title('average_distance_from_expected')
        plt.scatter(avg_balance['theta'],avg_balance['avg_dist'], c='r', label='all dominio')
        plt.scatter(avg_balance['theta'],avg_balance['avg_dist_in_eq_range'], c='k', label='in eq range')
        # plt.legend()
        plt.grid()


        plt.show()


# this function decide the amount of stock a trader wants to buy/sell (if any) 
# based on his current financial situation and prediction of the market trend
# prediction is in this case an extimated probability of market going up or down
def show_decision_functions(**d_kw):
    global def_D
    global def_B
    global def_SA
    
    D=lambda x: def_D(x,**d_kw)
    B=lambda x: def_B(x,**d_kw)
    SA=lambda x, f: def_SA(x, f, **d_kw)
    
    p=np.linspace(-1,1,100)
    P_N=lambda pr:1-D(pr)
    P_B=lambda pr:(D(pr)*(1 + B(pr)))/2
    P_S=lambda pr:(D(pr)*(1 - B(pr)))/2  
    
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title("ansatz functions")
    plt.plot(p,D(p), label='def_D')
    plt.plot(p,B(p), label='def_B')
    plt.plot(P_B(p),SA(p, P_B), label='def_SA')
    plt.ylim((-1,1))
    plt.xlim((-1,1))
    plt.grid()
    plt.legend()

    plt.subplot(1,2,2)
    plt.title("decision functions")
    plt.plot(p,P_N(p), 'c',label="nothing")
    plt.plot(p,P_B(p), 'g',label="buy")
    plt.plot(p,P_S(p), 'r',label="sell")
    plt.plot(p,P_B(p)*SA((p), P_B), 'g--', label="buy at market price")
    plt.plot(p,P_S(p)*SA((p), P_S), 'r--', label="sell at market price")
    plt.legend()
    plt.ylim((0,1))
    plt.xlim((-1,1))
    plt.grid()
    
    plt.show()
show_default_decision_functions=lambda: show_decision_functions(**default_parameter['decision'])

def show_sw_ansatz(**d_kw):
    x=np.linspace(0,1,101)
    
    if d_kw['sw_mode']=='beta':
        eta=d_kw['eta_coeff']

        for C in range(0,11):
            if d_kw['sw_coeff']=='eq_dist':
                C=C/10
                C_tr=d_kw['sw_coeff_postmap'](C, **d_kw)
                a=C_tr*eta
                b=(1-C_tr)*eta
        #         print(beta.pdf(x,a,b))
                plt.plot(x,beta.pdf(x,a,b), label="C={:.2f}; C_tr={:.2f}".format(C,C_tr))
            else:
                C=d_kw['sw_coeff']
                a=C*eta
                b=(1-C)*eta
                plt.plot(x,beta.pdf(x,a,b), label="C={:.2f}".format(C))
        
    if d_kw['sw_mode']=='uniform':
        plt.plot(x,np.ones(len(x)))
        
    plt.grid()
    plt.legend()
    plt.show()
show_default_sw_ansatz=lambda : show_sw_ansatz(**default_parameter['decision'])
    
    
########################################################################################### LAB

class Lab():
    """
    - can build new Market instances and performs simulation run
        - parametrically different mkt instances: param_sim_run
          (when initialized through a "market builder" (any function that returns Market instance)
        - same instances of one mkt instance  : multi_sim_run
          (when initialized through a market instance or with a market builder with set parameters)
    - keeps track on user provided metrics for these simulation, 
        - add metric(name_or_fn, ...)
    - can perform fit on metric results
        - fit_metric(fit_name, metric_name, fit_type, .... )
    - display stored info 
        - plot_metric
        - show_fit_results
        - show_fit_coeff
        - ...
    """
    
    _n_showfit=0 #<--- support variable for internal plotting management
    
    def __init__(self, market_or_builder, keep_all_market_iterations=False, **mkt_builder_param):
        """
        Can be called taking market instance or a function that returns a market instance
        """
        
        if isinstance(market_or_builder, Market):
            #MARKET DEEPCOPY makes the exact copy of given instance
            self.market=deepcopy(market_or_builder)
            self.market_builder=None
            self.mode='copy'
        else:
            #MARKET BUILDER makes different istance with same specification
            self.market=None
            self.market_builder=market_or_builder
            self.mode='build'
            
            self.param={}  #market in copy mode should not have these and must not use them
            self.set_param(**mkt_builder_param)
            
        self.keep_all_market_iterations=keep_all_market_iterations
        self.market_iterations={}
        
#         self._metricnames=[] #do I need both name and functions?
        self._getmetric_fn={} #since this is a dict of {name: fn} ?
        self.multi_sim_metric={}
        self.param_sim_metric={}
        
        self._add_default_metrics()
        
        self.clear_memory_at_each_run=True

    def set_param(self, **param_kw):
        if self.mode=='build':
            for name, val in param_kw.items():
                self.param[name]=val
        else:
            print(f" WARNING lab instance mode={self.mode}")
    
    def dir_param(self):
        if self.mode=='build':
            for name, value in self.param.items():
                print(f" {name}={value}")
        else:
            print(f" WARNING lab instance mode={self.mode}")
                          
    def get_new_mkt(self):
        if self.market_builder is not None:
            new_mkt=self.market_builder(**self.param)
        else:
            new_mkt=deepcopy(self.market)
        return new_mkt
        
    def _add_default_metrics(self):
        self.add_metric(lambda mkt: mkt.get_metric('daily_price','avg'), metric_name='avg_price')
#         self.add_required_result(lambda mkt: mkt.get_metric('avg_pred',partition='all'))
        
    def add_metric(self, name_or_fn, overwrite=False, metric_name=None, **getmetric_kw):
        """
        Add request of metric value to keep in memory when running market iterations.
        results can be retrieved by name_or_fn=name_string, 
            in which case the Lab will retrieve the data stored in market.metrics[name_string]
        Otherwise a callable can be passed as argument name_or_fn=given_function, 
              in which case the Lab will apply the callable at the end of iteration as given_function(mkt_obj). 
        name will be the provided name (in call by name), or the *code* name of the function (in call by function).
        Providing a keyword argument "metric_name" will overwrite any other name
        """
        # THE USE OF GETMETRIC_KW TO INSTANCE GETMETRIC_FN IS CURRENTLY NOT IMPLEMENTED
        # apart from the use in default name passing, when is used as 
        if isinstance(name_or_fn,str):
            name=name_or_fn
            getmetric_fn= lambda mkt: mkt.get_metric(name, **getmetric_kw)     
        elif hasattr(name_or_fn,'__call__'):
            name=name_or_fn.__code__.co_name
            getmetric_fn= name_or_fn
        if metric_name is not None:
            name=metric_name

        if name in self._getmetric_fn.keys():
            print(f"Warning, result {name} already required, through function ")
            if overwrite:
                print(" Overwriting older function")
            else:
                print(" call this function with overwrite=True")
                return
              
        self._getmetric_fn[name]=lambda mkt: getmetric_fn(mkt)
#         self._metricnames.append(name) 
            
        print(f"added required result {name}")

#----------------------------------------------------------------------- multi_run         
        
    def _reset_multi_sim_metric(self):
        for metric_name in self._getmetric_fn.keys():
            self.multi_sim_metric[metric_name]=[]
            
    def _init_multi_sim_metric(self): 
        if self.clear_memory_at_each_run:
            self._reset_multi_sim_metric()
            
    def _addata_multi_sim_metric(self, mkt_instance):
        for metric_name, getmetric_fn in self._getmetric_fn.items():
            self.multi_sim_metric[metric_name].append(getmetric_fn(mkt_instance))
        
    
    #iterations of same instance of market
    def multi_run(self, time, iterations, **ext_kw):
        """
        runs {iterations} iterations of same instance of market.
        Can work either with a lab that contains a Market instance (by deepcopying it)
        Or with a lab that contains a market builder (function that returns a Market instance)
            In which case the lab_obj.params must have been set (set_param(...))
        After each run, the lab will retrieve metric data by calling the function previously provied by add_metric(getmetric_fn))
            and store the metric results in self.multi_sim_metric[metric_name]=getmetric_fn(mkt_instance)
        """
        self._init_multi_sim_metric()
                
        for it in range(iterations):
            print("-"*50, f" run {it+1} of {iterations}")
            it_mkt=self.get_new_mkt()
            it_mkt.run(time, **ext_kw)
            
            if self.keep_all_market_iterations:
                self.market_iterations[it]=deepcopy(it_mkt)
                
            self._addata_multi_sim_metric(it_mkt)
            
#----------------------------------------------------------------------- param_run    
    def _reset_param_sim_metric(self):
        for metric_name in self._getmetric_fn.keys():
            self.param_sim_metric[metric_name]={}
            
    def _pardict_to_resdict(self, pardict):
        #TBI--- some checking of correct coordintion
        #returns a dictionary in the shape the lab works with
        return {parnames:{val:[] for val in pardict[parnames]} for parnames in pardict.keys() }
            
    def _init_param_sim_metric(self, param_names, param_values):
        pardict={param_names: param_values}
        if self.clear_memory_at_each_run:
            self._reset_param_sim_metric()
            for metric_name in self.param_sim_metric:
                self.param_sim_metric[metric_name]=self._pardict_to_resdict(pardict)
        else:
            print("consecutive runs not implemented")
         
    def _addata_param_sim_metric(self, mkt_instance, parnames, value):
        for metric_name, getmetric_fn in self._getmetric_fn.items():
            self.param_sim_metric[metric_name][parnames][value].append(getmetric_fn(mkt_instance))
    

    def param_run(self, time, param_names, param_values_list, iter_same_point=1, **ext_kw):
        """
        par_names is the names of the parameters which the market builder will take (MUST BE SAME NAME)
            param_values are the corresponding values
            param_names must be provided as string or iterable of strings
            param_values must be iterable of iterables of the values of the parameters
        when only a parameter is changing in simulation, call signature must be:
            param_run(365*5, 'theta', [0.1, 0.2, 0.4,.....], iter_same_point=... )
        when more parameter must change in simulation, call signature will be:
            param_run(365*5, ('theta','gamma'), [(0.1, -0.05),(0.2, 0.05),.....], iter_same_point=... )
        After each run, the lab will retrieve metric data by calling the function previously provied by add_metric(getmetric_fn))
            and store the metric results in 
            self.param_sim_metric[metric_name][param_names][param_value][iter_same_point_k]=getmetric_fn(mkt_instance)
            meaning param_sim_metric={metric_name:{param_name:{param_value:[ [],..., [subrun_result_k],... ,[]] } } }
            and can be retrieved by get_metric_result TBI?
            plot_param_sim_results, plot_results,...
        """
        assert self.market_builder!=None
        self._init_param_sim_metric(param_names, param_values_list)  
        N_tot_run=len(param_values_list)*iter_same_point
        C_tot_run=0 
        
        for param_values in param_values_list:

            #--------setting this as current builder parameters
            if isinstance(param_names, (list,tuple)):  #in case {(A_name, B_name):[(A_v1,B_v1),(A_v2,B_v2),...]}
                for name, value in zip(param_names, param_values): #meaning multiple parameters simultaneously
                    self.set_param(**{name:value})
            else:
                 self.set_param(**{param_names:param_values})
                    
            #--------subrun with these values
            for it in range(iter_same_point):
                C_tot_run+=1
                print("{:-<50}".format(f"run {C_tot_run} of {N_tot_run} total"), end='' )
                print("{:}={:} iteration {:} of {:}".format(param_names, param_values,it+1,iter_same_point) )
                it_mkt=self.get_new_mkt()
                it_mkt.run(time, **ext_kw)

                if self.keep_all_market_iterations:  #not tested
                    self.market_iterations[it]=deepcopy(it_mkt)
                    
            #-------- adding metric data to instance memory
                self._addata_param_sim_metric(it_mkt, param_names, param_values)       
    
#     #functional run taking a generator of parameters
#     def funct_run(self, time, param_generator, iter_same_point=1, **ext_kw):
#         assert self.market_builder!=None
#         print("to_be_implemented")


#----------------------------------------------------------------------- fitting   

    def fit_metric(self, metric_name, fit_type, start_slice=None, end_slice=None, resolution=1, poly_order=None, **fit_kw):
        """
        Currently implemented only for parametric runs (2.9.0)
        Takes all simulation metric results, kept in memory as 'metric_name', then apply a fit of 'fit_type'.
           fit_type can be 'lin', 'exp' or 'pol' (the latter requires the user to provide poly_order>0)
        The fit is applied in a window of [start_slice:end_slice:resolution],
           meaning the values that the program will be trying to fit are those between [start,end[ with delta of resolution
           end_slice is the last value *not included* in fit dominio, and can be left None for last value of simulation
        The fit is weighted on y**weight_coeff, which can be provided in kwargs and is 2 by default
           this mechanism works in each of the different fit_type, 
           whether the provided metric is 1-D (one value per day) or 2-D (array of value per day)
        User can provide a name by kwarg fit_name
            which is the name used by lab instance for storing the results
            and can be used by user to retrieve fit results
        After the fit is succesfull, the function store its results in self.all_fit_results[fit_name].
           all_fit_results={fit_name: {dominio:[], results: {param_names: {param_values: [{function:[], coeff:[]}] }}}}
           where ...['function'] is the function that fits the data and ...['coeff'] the respective coefficients
        The fit results can be later taken from lab instance through lab_obj.get_fit_results TBI, 
            get_fit_coeff, show_fit_results, fit_summary...
        The storing mechanism differentiate:
            differently named fit
            equally named fit applyed on different metric
            *equally named fit applyed on same metric with different dominio and/or different fit parameters (TBI)
            everything else is replaced by name (currently also *) and user is given a warning
        If no name is provided by the user, the assigned name will be {fit_type}_{start}-{end}          
        """
        if not len(self.param_sim_metric) and len(self.multi_sim_metric):
    #         result_data=deepcopy(self.multi_results)
            print("fitting of multi_results not yet implemented")
            return
        elif len(self.param_sim_metric) and not len(self.multi_sim_metric):
            pass #only implemented 
        else:
            print("ERROR: no metric in this lab instance or both multi and param are non-None")
            return
        metric_data=self.param_sim_metric[metric_name]
        fit_result={metric_name:deepcopy(metric_data)}

    #     self.param_results={result_name: param_names: param_values: [subran_results]}
        for param_names, param_values_dict in metric_data.items():
    #         print(param_names)
    #         print("-"*60)
            for param_value, array_of_subrun_data in tqdm(param_values_dict.items(), desc=f"{fit_type} fitting {metric_name}"):
    #             print(param_names, param_value)
    #             print("-"*60)
                for sr_k, subrun_data in enumerate(array_of_subrun_data):

                    if start_slice is None:
                        start_slice_N  =0
                    else:
                        start_slice_N =start_slice
                    if end_slice is None:
                        end_slice_N    =len(subrun_data) 
                    else:
                        if end_slice<0:
                            end_slice_N=len(subrun_data)+end_slice
                        else:
                            end_slice_N    =end_slice 
    #                 end_slice_N+=1 #<---- IF I WANT TO INCLUDE LAST ELEMENT
    
                    dominio=np.array(list(range(start_slice_N, end_slice_N, resolution)))

                    y=np.array([subrun_data[d] for d in dominio])
                    x=np.array([i for i in range(len(dominio))])
                    if hasattr(y[0], '__len__') and len(y[0])>1: #<--- if daily metric value is an array  (metric was N_day*N_ag) 
                        len_y=len(y[0])
                        x=np.array([[x_i]*len_y for x_i in x])
                    else:
                        len_y=1
                    y=y.flatten()
                    x=x.flatten()
    #                 return x,y
    #                 print(x)
    #                 print(y)

                    weight_coeff=fit_kw.get('weight_coeff', 2)
                    if 'exp' in fit_type.lower():
                        A,B=np.polyfit(x, np.log(y), 1, w=y**weight_coeff)
    #                     print(A,B)
                        fitted_fn=np.exp(B)*np.exp(x*A)
#                         fitted_fn=np.array([fitted_fn[f_v] for f_v in range(0, len(fitted_fn), len_y)])
                        fitted_fn=fitted_fn[::len_y]
                        coeff=(A,B)
                    if 'pol' in fit_type.lower():
                        C=np.polyfit(x, y, poly_order, w=y**weight_coeff)
                        fitted_fn=np.array([C[poly_order-i]*x**i for i in range(poly_order+1) ]).sum(axis=0)
#                         fitted_fn=np.array([fitted_fn[f_v] for f_v in range(0, len(fitted_fn), len_y)])
                        fitted_fn=fitted_fn[::len_y]
                        coeff=np.array(C)
                    if 'lin' in fit_type.lower():
                        C=np.polyfit(x, y, 1         , w=y**weight_coeff)
                        fitted_fn=np.array([C[1         -i]*x**i for i in range(1         +1) ]).sum(axis=0)
#                         fitted_fn=np.array([fitted_fn[f_v] for f_v in range(0, len(fitted_fn), len_y)])
                        fitted_fn=fitted_fn[::len_y]
                        coeff=np.array(C)
                    fit_result[metric_name][param_names][param_value][sr_k]={'function':fitted_fn, 'coeff':coeff}

        fit_name=fit_kw.get('fit_name')
        if fit_name is None:
            fit_name=fit_type+'_fit_'+str(start_slice_N)+'-'+str(end_slice_N)
        if hasattr(self, 'all_fit_results'):
            if fit_name in self.all_fit_results:
    #           case of same fitname on same dominio
                if np.array([dominio== self.all_fit_results[fit_name]['dominio']]).all():#<--- meaning same fit characteristics
                    if metric_name in self.all_fit_results[fit_name]['results']:#<--- meaning fit on that metric already executed before
                        print(f"replacing previous results for {fit_name} of {metric_name}")       
                    self.all_fit_results[fit_name]['results'][metric_name]=fit_result[metric_name]
    #           case of same fitname on different dominio
                else:
                    print("WARNING NOT PROPERLY IMPLEMENTED replacing previous")
                    fit_name+=str(start_slice_N)+'-'+str(end_slice_N)    #<---- BETA BETA never used
                     #changes the name so should not override it 
                     #unless blablabla but not important atm
    #                 self.all_fit_results[fit_name]={'dominio':dominio,'results':fit_result}
                    self.all_fit_results[fit_name]={'dominio':dominio,'results':fit_result}
            else:
                self.all_fit_results[fit_name]={'dominio':dominio,'results':fit_result}
        else:
            self.all_fit_results={fit_name:{'dominio':dominio,'results':fit_result}}
    #     all_fit_results={fit_name: {dominio:[], results: {param_names: {param_values: [{function:[], coeff:[]}] }}}}
    #     all_fit_results['exp']['results'][('theta','gamma')][(0.5,0.8)][0]['function']
    
#----------------------------------------------------------------------- getmetrics (minimal implementation)
    def get_multi_metric(self, metric_name):
        return self.multi_sim_metric[metric_name]

    def get_param_metric(self, metric_name, *more_needed):
        print("WARNING: get metric not implemented for mode PARAM")
        return

    def get_metric(self, metric_name, *args, **kwargs):
        if not len(self.param_sim_metric) and len(self.multi_sim_metric):
            return get_multi_metric(self, metric_name)
        elif len(self.param_sim_metric) and not len(self.multi_sim_metric):
            return get_param_metric(self, metric_name, *args, **kwargs)
        else:
            print(" both multi results and param results are present: specify which to get")
            return
    
    
#----------------------------------------------------------------------- plotting   
                        
    def plot_multi_sim_metric(self, metric_name, start=0, end=-1, alpha=0.5):
        
        data=self.multi_sim_metric[metric_name]
        if end<0:
            end=len(data[0])+end
        end+=1
        
        for run_data in data:
            plt.plot(run_data[start:end], c='k', alpha=alpha)
        plt.title(metric_name)
        
    def plot_param_sim_metric(self, metric_name, start=0, end=-1, legend_loc='best'):
        end+=1
        colors=['r','g', 'b','y','m','c','k']*5
        
        for par_name in self.param_sim_metric[metric_name]:
            for color, par_val in zip(colors, self.param_results[metric_name][par_name]):      
                #<--- not tested (suspect param_results deprecated. New name is param_sim_metric)
                for subiter, subrun_data in enumerate(self.param_results[metric_name][par_name][par_val]):
                    if subiter==0:this_label=f"{metric_name}={par_val}"
                    else:         this_label=None
                    if end==0:  #which means the call with end=-1
                        plt.plot(subrun_data[start:   ], label=this_label, c=color)
                    else:
                        plt.plot(subrun_data[start:end], label=this_label, c=color)

        if legend_loc!='off':
            plt.legend(loc=legend_loc)
        plt.grid()
        plt.title(metric_name)
        
    def plot_metric(self, metric_name, start=0, end=-1, legend_loc='best'):
        if   not len(self.param_sim_metric) and len(self.multi_sim_metric):
            parametric_plot=False
        elif len(self.param_sim_metric) and not len(self.multi_sim_metric):
            parametric_plot=True
        else:
            print(" both multi results and param results are present: specify which to display")
            return
            
        if not parametric_plot:
            self.plot_multi_sim_metric(metric_name, start=start, end=end)
        else:
            self.plot_param_sim_metric(metric_name, start=start, end=end, legend_loc=legend_loc)
            
    def plot_results_density(self, metric_name):
        print("to be implemented")


    def plot_fit_metric(self, fit_name, metric_name, 
                        plot_grid=True, n_columns=5, height_row=4, same_y_scale=True, legend_loc='lower right', fit_col='r'):
        if not hasattr(self, 'all_fit_results'):
            print(" no fit has yet been performed")
            return
        """
        plot_grid=True means results are divided by run parameter in a grid of subplots; 
        plot_grid=False means all results are plotted in the same graph
        same_y_scale=True set the same y_min and y_max for every subplotlegend_loc='lower right', 
        before calling this function, the respective lab.fit(..., metric_name, fit_name=fit_name) must have been called,
        otherwise this function raise an error or warns the user that no fit has been performed
        """
        dominio=self.all_fit_results[fit_name]['dominio']
        fit_res=self.all_fit_results[fit_name]['results'][metric_name]
        sim_mtr=self.param_sim_metric[metric_name]
        fit_coeff=deepcopy(self.param_sim_metric[metric_name])
    
        #------------checking for tot number of graphs and (if needed) y_max and y_min
        tot_graph=0
        if same_y_scale:
            min_y=+2**64
            max_y=-2**64
        for _, param_values_dict in sim_mtr.items():
            for _, array_of_subrun_data in param_values_dict.items():
                for _, subrun_data in enumerate(array_of_subrun_data):
                    tot_graph+=1
                    if same_y_scale:
                        min_y=min(min_y, min(np.array(subrun_data).flatten()))
                        max_y=max(max_y, max(np.array(subrun_data).flatten()))
        if same_y_scale:
            min_y*=(1+0.05)
            max_y*=(1+0.05)
            
        if plot_grid:                
            n_rows=int(tot_graph//n_columns+np.ceil(tot_graph%n_columns))
    #         plt.figure(figsize=(18, n_rows*height_row)) #<--- not needed if this function is called by show_fit_results
        else:
            plt.title("all fit results")
            plt.grid(visible=True)
        this_graph=0
        for param_names, param_values_dict in sim_mtr.items():
            for param_values, array_of_subrun_data in param_values_dict.items():
                this_graph+=1
                
                if plot_grid: #create subplot title
                    plt.subplot(n_rows, n_columns, this_graph)
                    plt.grid(visible=True)
                    if isinstance(param_names, (list,tuple)):
                        plt.title(";".join([f"{p_n}"[:5]+f" {p_v}"[:5] for p_n, p_v in zip(param_names, param_values)]))
                    else:
                        plt.title(f"{param_names}"[:10]+f"{param_values:.5f}")
                        
                #plotting for every subrun data        
                for sr_k, subrun_data in enumerate(array_of_subrun_data):
                    sr_label= lambda text, sr_k, is_fitfn: text if (not is_fitfn and not sr_k and not Lab._n_showfit) or (is_fitfn and not sr_k) else None

                    true_fn =sim_mtr[param_names][param_values][sr_k]
                    fit_fn  =fit_res[param_names][param_values][sr_k]['function']
    #                 fit_coeff[param_names][param_values][sr_k]=fit_res[param_names][param_values][sr_k]['coeff']
                    
                    #dominio is the array of value for the fit  (can have a resolution of n days!=1)
                    #    while dominio_interval will be the whole window where that fit was performed
                    dominio_interval=list(range(dominio[0],dominio[-1]))
                    
                    #------ case for true_fn iterable of iterables (if daily data is anarray)
                    #       in which case a scatter plot + avg is required
                    if hasattr(true_fn[0], '__len__') and len(true_fn[0])>1:    
                        len_y=len(true_fn[0])
                        plt.scatter(np.array([[d_v]*len_y for d_v in dominio_interval]).flatten(),
                                    np.array([true_fn[d_v] for d_v in dominio_interval]).flatten(),
                                    c='c', alpha=1/len_y+0.001,
                                   label=sr_label('all true values' ,sr_k, False ))
                        plt.plot(dominio_interval, [np.array(true_fn[x]).mean() for x in dominio_interval],
                                c='k', linewidth=1.5, alpha=0.8,
                                label=sr_label('avg true values' ,sr_k, False))
                    #------ case for true_fn iterable of values (if daily data is single-value)
                    #       in which case a basic plot is required
                    else:
                        plt.plot(dominio_interval, [true_fn[x] for x in dominio_interval], 
                                 c='k', linewidth=1.5, alpha=0.7,
                                label=sr_label( 'true fn',sr_k, False))


                    #------ plotting fit_fn
                    plt.plot(dominio, fit_fn, 
                             c=fit_col,  linewidth=1.5, alpha=0.8,
                            label=sr_label( 'fit fn: '+fit_name ,sr_k, True))
                    
                    if same_y_scale:
                        plt.ylim([min_y,max_y])
                        
                if legend_loc.lower()!='off':
                    plt.legend(loc=legend_loc)   
    #     plt.show()          


    def print_fit_coeff(self, fit_names, metric_name):
        if not hasattr(self, 'all_fit_results'):
            print(" no fit has yet been performed")
            return
    #     dominio=self.all_fit_results[fit_name]['dominio']
    #     fit_res=self.all_fit_results[fit_name]['results'][metric_name]
        sim_mtr=self.param_sim_metric[metric_name]
    #     fit_coeff={fit_name: self.all_fit_results[fit_name]['results'][metric_name] for fit_name in fit_names}
        if not isinstance(fit_names,(tuple,list)):
            fit_names=[fit_names]

        for param_names, param_values_dict in sim_mtr.items():
            for param_values, array_of_subrun_data in param_values_dict.items():
                if isinstance(param_names, (list,tuple)):
                    print(";".join([f"{p_n}"[:5]+f" {p_v}"[:5] for p_n, p_v in zip(param_names, param_values)])+'-'*30)
                else:
                    print(f"{param_names}"[:10]+f"={param_values:.5f}"+'-'*30)
                for sr_k, subrun_data in enumerate(array_of_subrun_data):
                    if len(array_of_subrun_data)>1:
                        print(f"    subrun {sr_k}: ")
                    for fit_name in fit_names:
                        print(f"        {fit_name:<15}: ", end='')
                        this_fit_coeff=self.all_fit_results[fit_name]['results'][metric_name][param_names][param_values][sr_k]['coeff']
                        for coeff_k, coeff in enumerate(this_fit_coeff):
                            print(f" C[{coeff_k:2}]={coeff:>10.5f}", end='; ')
                        print()
                        
    def get_fit_coeff(self, fit_name, metric_name):
        """
        returns a {param_values: [[],...[k-st fit coeff values]]  }
        """
        if not hasattr(self, 'all_fit_results'):
            print(" no fit has yet been performed")
            return
        coeff_data_dict=self.all_fit_results[fit_name]['results'][metric_name]
        coeff_data={}
        for param_names, param_values_dict in coeff_data_dict.items():
            if isinstance( param_names, (list,tuple)) and len(param_names)>1:
                print("N-Dimension fit coeff get not implemented")
                return
            for param_values, subrun_data_array in param_values_dict.items():
                coeff_data[param_values]=[]
                for subrun_data in subrun_data_array:
                    coeffs=subrun_data['coeff']
                    N_coeff=len(coeffs)
                    for k, coeff in enumerate(coeffs):
                        while len(coeff_data[param_values])<k+1:
                            coeff_data[param_values].append([])
                        coeff_data[param_values][k].append(coeff)#<--- value of k-st coefficient for simulation parameter param_value
            return coeff_data #{param_values: [[],...[k-st coeff values]]   
        
        
    def show_fit_coeff(self, fit_name, metric_name, coeff_names=None, n_col='auto', y_k_mapping='def', omitt=[], mode='distr'):
        "mode can be distr (avg + errorbars) or scatter (every point is plotted)"
        if not hasattr(self, 'all_fit_results'):
            print(" no fit has yet been performed")
            return
        coeff_data_dict=self.all_fit_results[fit_name]['results'][metric_name]
    #     coeff_data={p_n:{p_v:[subrun_dict {... 'coeff':[coeff values]}
        coeff_data_xy=[{'x':[],'y':[]}]
        if y_k_mapping is not None:
            if y_k_mapping == 'def':
                y_k_mapping=lambda y, k: y
                if 'exp' in fit_name:
#                     print('using def exp mapping')
                    y_k_mapping=lambda y, k: np.exp(y) if k==1 else -1/y 


        for param_names, param_values_dict in coeff_data_dict.items():
            for param_values, subrun_data_array in param_values_dict.items():

                include_values=False   #<--------- checking if values are not to be omitted
                if len(omitt)==0:
                    include_values=True
                else:
                    for p_omitt in omitt:
                        for v_p_omitt, v_p in zip(list(p_omitt), param_values if hasattr(param_values, '__iter__') else [param_values]):
                            if v_p_omitt!=v_p and v_p_omitt!='any':
                                include_values=True

                if not include_values:
                    continue

                for subrun_data in subrun_data_array:
                    coeffs=subrun_data['coeff']
                    N_coeff=len(coeffs)
                    for k, coeff in enumerate(coeffs):
                        while len(coeff_data_xy)<k+1:
                            coeff_data_xy.append({'x':[],'y':[]})
                        if isinstance( param_names, (list,tuple)) and len(param_names)>1: 
                            coeff_data_xy[k]['x'].append(param_values[0])
                            #<--- when there is more than one parameter, only the first is represented in x axis
                            x_title=param_names[0]
                        else:
                            coeff_data_xy[k]['x'].append(param_values)
                            x_title=param_names

                        coeff_data_xy[k]['y'].append(coeff)

            # -------- NB still inside coeff_data_dict loop (should not make any difference in actual use-cases)

            if coeff_names is None:
                coeff_names=['coeff_'+str(k) for k in range(N_coeff)]
            if len(coeff_names)!=N_coeff:
                raise ValueError(f"{len(coeff_names)=} must be == {N_coeff=}")
            n_tot_graphs=N_coeff
            if n_col=='auto':
                n_col=min(5,N_coeff)
            n_row=n_tot_graphs//n_col+int(np.ceil(n_tot_graphs%n_col))

            plt.figure(figsize=(18,n_row*6))
            for k, (this_coeff_data_xy, this_coeff_name) in enumerate(zip(coeff_data_xy, coeff_names)):
                plt.subplot(n_row, n_col, k+1)
                plt.grid()
                plt.title(fit_name+" "+this_coeff_name)
                x=np.array([x_k for x_k in this_coeff_data_xy['x']])
                plt.xlabel(x_title)
                y=np.array([y_k for y_k in this_coeff_data_xy['y']])
                if y_k_mapping is not None:
                    y=y_k_mapping(y,k)  #any more complex operation should require a get_coeff and external op
                if 'scat' in mode.lower():
                    plt.scatter(x,y, color='k')
                if 'dist' in mode.lower():
                    x2y={}
                    for t_x, t_y in zip(x,y):
                        if t_x not in x2y:
                            x2y[t_x]=[]
                        x2y[t_x].append(t_y)
                    for x_c in x2y.keys():
                        x2y[x_c]=np.array(x2y[x_c])

                    x_,y_avg, y_err=[],[],[]
                    for x_c, y_c in x2y.items():
                        x_.append(x_c)
                        y_avg.append(y_c.mean())
                        y_err.append(y_c.std()/2 )
                    plt.scatter(x_, y_avg, color='k', marker='x', label='avg')
                    plt.errorbar(x_, y_avg, y_err, ecolor='r', linewidth=0, elinewidth=2., capsize=4, label='st. dev')
                    plt.legend()
            plt.show()


    def show_fit_results(self, fit_names, results, 
                         plot_grid=True, n_columns=5, height_row=4, same_y_scale=True, legend_loc='lower right'):
        if not isinstance(fit_names,(tuple,list)):
            fit_names=[fit_names]

        sim_mtr=self.param_sim_metric[results]
        tot_graph=0
        for param_names, param_values_dict in sim_mtr.items():
            for param_values, array_of_subrun_data in param_values_dict.items():
                for sr_k, subrun_data in enumerate(array_of_subrun_data):
                    tot_graph+=1
        if plot_grid:                
            n_rows=int(tot_graph//n_columns+np.ceil(tot_graph%n_columns))
            plt.figure(figsize=(18, n_rows*height_row)) 

        colors=['r','g','m','y','b','c']
        for n, (fit_name,col) in enumerate(zip(fit_names,colors)):
            Lab._n_showfit=n
            self.plot_fit_metric(fit_name, results, plot_grid, n_columns, height_row, same_y_scale, legend_loc, fit_col=col)
        plt.show()
        Lab._n_showfit=0
        self.print_fit_coeff(fit_names, results)

    def print_mtr_summary(self, mode='auto'):
        if mode=='auto':
            if   not len(self.param_sim_metric) and len(self.multi_sim_metric):
                metric_data=self.multi_sim_metric
                mode='multi'
            elif len(self.param_sim_metric) and not len(self.multi_sim_metric):
                metric_data=self.param_sim_metric
                mode='param'
        print()
        print()
        print("-"*50+" METRICS INFO")
        print()
        if mode=='param':
            for metric_name in self._getmetric_fn.keys():
                print("-"*10+"metric name:", metric_name)
                if metric_name not in metric_data:
                    print(" "*10+" "*13+"no run yet executed")
                else:
                    print(" "*10+"performed on parameters:")
                    for param_names, param_values_dict in metric_data[metric_name].items():
                        print(" "*10+" "*5+str(param_names)+"="+",".join([f"{p_v}" for p_v in param_values_dict.keys()]) )
        else:
            print("MODE", mode, "NOT IMPLEMENTED")

    def print_fit_summary(self):
        print()
        print()
        print("#"*50+" FITS INFO")
        print()
        if not hasattr(self, 'all_fit_results'):
            print("######### no fit has yet been performed")

        for fit_name, fit_data in self.all_fit_results.items():
            print("######### fit name:", fit_name)
            dominio= fit_data['dominio']
            print("          dominio from", dominio[0], "to (included)", dominio[-1],"of size", len(dominio))
            print("          performed on metrics:", end='')
            for k, metric_name in enumerate(fit_data['results']):
                print("                               "*int(bool(k)), metric_name)





how_to_use="""
HOW TO USE: 

NB UPDATED TO 1.7 (ALL OF THIS IS DEPRECATED AND NO LONGER TRUE)

INITIALIZE A BOOK
bb=Book()

INITIALIZE MARKET
market=Market(N tot agents, book they will act on (1) )

SET MARKET AGENTS PARTITION
market.make_agents_partition({'name_part1':size,... , 'name_partn':'auto'})
   size can be int, float (percentage) or 'auto' (remaining agents)

SET PARTITION-SPECIFIC PARAMETERS 
market.give_part_kwargs(partition_name, parameters I want the agents in that partition to store internally)
   this is to be used if some agent partition has his own parameter that they will use in the process

DEFINE FUNCTION FOR AGENTS TO EXECUTE
def fn(self, **kwargs)
   self at the beginning and **kwargs at the end is required!!!
   it will recieve arguments from market.step
   which must be acquired with **kward dictionary
   like  ext_par=kwargs['ext_par']
   it is allowed to act as if it is defined in the agent class itself, and call self. 
   so it can acquire agent specific parameters and modify internal variables
   like  par=self.my_pers_kwargs['par'] (which are given in step before)
   like  self.state['sentiment']= ... 

SET PARTITIONS-SPECIFIC FUNCTION
market.set_part_fn(partition_name, what "mental" step it is referred, the function to call)
   this means all agent of that partition will call that function during "what" process of their decision
   it will eventually override a preexistent function if present (no warning)

DEFINE PRE AND POST MARKET SPECIAL FUNCTION
def pre_market_fn(self, **kwargs)
   PRE will operate AT THE BEGINNING of the market day, POST will at the end (after metrics are evaluated)
   might be used for evaluating some env condition or setting them

SET PRE AND POST MARKET SPECIAL FUNCTION
market.set_spec_fn(when, fn)
   when must be 'pre' or 'post', fn is the function that will be called 'when'

DEFINE FUNCTIONS FOR METRICS EVALUATION
metrics will be evaluated in 3 different times of the process: pre step, inside step, post step
will therefore need a step-initializer function, 
a function able take parameter from inside the for-loop on the agents
a function to call at the end of each step to assign step-final value o metric evaluation
all these functions are called during each steps
each of these must be defined as def metric_step_ev(self, metric_obj, **kwargs)
where self is an instance of market, **kwargs are external parameters
and metric_obj is the metric variable that they want to modify (usually an array)
which is contained in the market instance calling it 
example of definition is:
def my_avg(self, metr_obj, **kwargs):
   time=self.time
   metr_obj[-1].append(time* blablabla)
NB need to be careful to assign these to the proper step-time:

SET FUNCTION FOR METRICS EVALUATION
market.add_metric(metric_name, fn, fn_init=None, fn_step_init=None, fn_inside=None, **init_kw)
  fn_init only called at the beginning of this metric life, when creating the metric object. Use init_kw if needed
  fn_step_init will be called at the beginning of each step
    by default is just append(0)
  fn_inside will be called *inside* the for_ agents loop
    by default does nothing
  fn is called at the end of each step
    must be defined as indicated above


RUN THE MARKET
market.step(all parameters/variables that the agent/special/metric function will seek)
   if some parameter is needed by agents during execution, it must be given to step(...)
   if some parameter is time dependent or inline evaluated externally, it must be given to step(...)
"""


def how_to():
    print(how_to_use)
    