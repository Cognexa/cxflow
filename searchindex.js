Search.setIndex({docnames:["advanced/config","advanced/dataset","advanced/hook","advanced/index","advanced/main_loop","advanced/model","cli","cxflow/cxflow","cxflow/cxflow.constants","cxflow/cxflow.datasets","cxflow/cxflow.hooks","cxflow/cxflow.models","cxflow/cxflow.utils","cxflow/cxflow.utils.config","cxflow/cxflow.utils.misc","cxflow/cxflow.utils.profile","cxflow/cxflow.utils.reflection","cxflow/index","getting_started","index","tutorial"],envversion:51,filenames:["advanced/config.rst","advanced/dataset.rst","advanced/hook.rst","advanced/index.rst","advanced/main_loop.rst","advanced/model.rst","cli.rst","cxflow/cxflow.rst","cxflow/cxflow.constants.rst","cxflow/cxflow.datasets.rst","cxflow/cxflow.hooks.rst","cxflow/cxflow.models.rst","cxflow/cxflow.utils.rst","cxflow/cxflow.utils.config.rst","cxflow/cxflow.utils.misc.rst","cxflow/cxflow.utils.profile.rst","cxflow/cxflow.utils.reflection.rst","cxflow/index.rst","getting_started.rst","index.rst","tutorial.rst"],objects:{"":{cxflow:[7,0,0,"-"]},"cxflow.MainLoop":{PREDICT_STREAM:[7,2,1,""],TRAIN_STREAM:[7,2,1,""],UNUSED_SOURCE_ACTIONS:[7,2,1,""],__init__:[7,3,1,""],_check_sources:[7,3,1,""],_create_epoch_data:[7,3,1,""],_run_epoch:[7,3,1,""],_run_zeroth_epoch:[7,3,1,""],evaluate_stream:[7,3,1,""],get_stream:[7,3,1,""],run_prediction:[7,3,1,""],run_training:[7,3,1,""],train_by_stream:[7,3,1,""]},"cxflow.constants":{CXF_CONFIG_FILE:[8,4,1,""],CXF_FULL_DATE_FORMAT:[8,4,1,""],CXF_HOOKS_MODULE:[8,4,1,""],CXF_LOG_DATE_FORMAT:[8,4,1,""],CXF_LOG_FILE:[8,4,1,""],CXF_LOG_FORMAT:[8,4,1,""]},"cxflow.datasets":{AbstractDataset:[9,1,1,""],BaseDataset:[9,1,1,""]},"cxflow.datasets.AbstractDataset":{__init__:[9,3,1,""]},"cxflow.datasets.BaseDataset":{__init__:[9,3,1,""],_init_with_kwargs:[9,3,1,""]},"cxflow.hooks":{AbstractHook:[10,1,1,""],AccumulateVariables:[10,1,1,""],CatchSigint:[10,1,1,""],Check:[10,1,1,""],ComputeStats:[10,1,1,""],LogProfile:[10,1,1,""],LogVariables:[10,1,1,""],SaveBest:[10,1,1,""],SaveEvery:[10,1,1,""],StopAfter:[10,1,1,""],TrainingTerminated:[10,6,1,""],WriteCSV:[10,1,1,""]},"cxflow.hooks.AbstractHook":{__init__:[10,3,1,""],after_batch:[10,3,1,""],after_epoch:[10,3,1,""],after_epoch_profile:[10,3,1,""],after_training:[10,3,1,""],before_training:[10,3,1,""]},"cxflow.hooks.AccumulateVariables":{__init__:[10,3,1,""],_reset_accumulator:[10,3,1,""],after_batch:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.hooks.CatchSigint":{__init__:[10,3,1,""],_sigint_handler:[10,3,1,""],after_batch:[10,3,1,""],after_training:[10,3,1,""],before_training:[10,3,1,""]},"cxflow.hooks.Check":{__init__:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.hooks.ComputeStats":{AGGREGATIONS:[10,2,1,""],__init__:[10,3,1,""],_compute_aggregation:[10,5,1,""],_save_stats:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.hooks.LogProfile":{after_epoch_profile:[10,3,1,""]},"cxflow.hooks.LogVariables":{UNKNOWN_TYPE_ACTIONS:[10,2,1,""],__init__:[10,3,1,""],_log_variables:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.hooks.SaveBest":{OBJECTIVES:[10,2,1,""],__init__:[10,3,1,""],_get_value:[10,3,1,""],_is_value_better:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.hooks.SaveEvery":{SAVE_FAILURE_ACTIONS:[10,2,1,""],__init__:[10,3,1,""],after_epoch:[10,3,1,""],save_model:[10,5,1,""]},"cxflow.hooks.StopAfter":{__init__:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.hooks.WriteCSV":{MISSING_VARIABLE_ACTIONS:[10,2,1,""],UNKNOWN_TYPE_ACTIONS:[10,2,1,""],__init__:[10,3,1,""],_write_header:[10,3,1,""],_write_row:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.models":{AbstractModel:[11,1,1,""]},"cxflow.models.AbstractModel":{__init__:[11,3,1,""],input_names:[11,2,1,""],output_names:[11,2,1,""],restore_fallback:[11,2,1,""],run:[11,3,1,""],save:[11,3,1,""]},"cxflow.utils":{config:[13,0,0,"-"],misc:[14,0,0,"-"],profile:[15,0,0,"-"],reflection:[16,0,0,"-"]},"cxflow.utils.config":{config_to_file:[13,7,1,""],config_to_str:[13,7,1,""],load_config:[13,7,1,""],parse_arg:[13,7,1,""]},"cxflow.utils.misc":{DisabledLogger:[14,1,1,""],DisabledPrint:[14,1,1,""]},"cxflow.utils.misc.DisabledLogger":{__enter__:[14,3,1,""],__exit__:[14,3,1,""]},"cxflow.utils.misc.DisabledPrint":{__enter__:[14,3,1,""],__exit__:[14,3,1,""]},"cxflow.utils.profile":{Timer:[15,1,1,""]},"cxflow.utils.profile.Timer":{__enter__:[15,3,1,""],__exit__:[15,3,1,""],__init__:[15,3,1,""]},"cxflow.utils.reflection":{create_object:[16,7,1,""],find_class_module:[16,7,1,""],get_class_module:[16,7,1,""],list_submodules:[16,7,1,""],parse_fully_qualified_name:[16,7,1,""]},cxflow:{MainLoop:[7,1,1,""],constants:[8,0,0,"-"],datasets:[9,0,0,"-"],hooks:[10,0,0,"-"],models:[11,0,0,"-"],utils:[12,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","data","Python data"],"5":["py","staticmethod","Python static method"],"6":["py","exception","Python exception"],"7":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:data","5":"py:staticmethod","6":"py:exception","7":"py:function"},terms:{"06d":8,"0th":7,"10th":10,"15s":8,"1st":2,"2nd":2,"800x600":1,"800x600x3":1,"abstract":11,"boolean":[4,5],"case":[0,1,2,5,10,20],"catch":10,"class":[0,1,2,5,16,20],"default":[0,4,6,10],"final":[0,1,4,18,20],"float":[10,15],"function":[1,4,7,10,14,20],"import":[0,1,2,5,8,16,20],"int":[1,2,7,10,20],"new":[0,2,5,9,10,15,20],"null":14,"return":[4,5,7,9,10,11,13,16,20],"short":2,"static":10,"super":2,"switch":10,"true":[0,1,5,7,11],"try":20,"var":0,"while":[0,4,5],For:[0,1,2,5,10,20],One:[4,10],THE:1,That:[1,20],The:[0,2,3,4,5,6,9,10,18,19,20],There:4,These:[0,5],With:[6,19],__enter__:[14,15],__exit__:[14,15],__init__:[2,7,9,10,11,15],_accumul:10,_check_sourc:7,_compute_aggreg:10,_create_epoch_data:7,_create_model:20,_epoch_limit:[2,10],_get_valu:10,_init_with_kwarg:[1,9,20],_is_value_bett:10,_log_vari:10,_name:15,_on_missing_vari:10,_on_unknown_typ:10,_on_unused_sourc:7,_profil:15,_reset_accumul:10,_run_epoch:7,_run_zeroth_epoch:7,_save_every_n_epoch:10,_save_stat:10,_session:20,_sigint_handl:10,_start:15,_stream:[1,9,10,20],_variabl:10,_write_head:10,_write_row:10,abl:[1,5,15,16,20],about:[5,10,16],abov:[0,1,2,5,10,20],abstract_hook:[2,10],abstractdataset:[7,9,11,20],abstracthook:[2,7,10],abstractmodel:[5,7,10,11],accept:[1,2,5,20],access:[6,20],accord:[11,20],accumul:[2,10],accumulate_vari:10,accumulatevari:[2,10],accuraci:[0,2,10,20],act:2,action:[7,10,14,20],actual:[0,1,2,6],adam:20,adamoptim:20,add:[9,20],adding:20,addit:[0,2,3,4,7,9,10,11,13,18,20],addition:[5,9,10],additional_arg:13,advanc:[0,19,20],advantag:0,after:[0,2,4,10,20],after_batch:[3,7,10],after_batch_hooks_train:10,after_epoch:[3,7,10],after_epoch_hook:10,after_epoch_profil:[2,7,10],after_train:[2,7,10],afterward:4,again:[0,20],aggreag:10,aggreg:10,algorithm:20,all:[0,1,2,5,6,7,9,10,11,16,20],allow:[2,6,10,15,20],almost:0,along:[1,4],alreadi:[0,1,2,20],also:[2,4,5,11,20],altern:4,although:[5,6],altogeth:2,among:10,analog:[0,1],anchor:0,ani:[1,2,5,7,9,10,13,14,16,20],anim:[0,1,5],annoi:1,anoth:0,answer:20,anyth:20,anywai:10,api:[6,9,10,19],append:[11,15],approach:0,approch:0,appropri:0,arbitrari:[1,20],arbitrarili:0,aren:0,arg1:2,arg2:2,arg:[2,13,14,15,16],argmax:20,argument:[0,1,2,4,5,10,13,20],around:20,artifact:20,artifici:1,asctim:8,assum:1,attempt:5,attribut:[0,20],attributeerror:7,augment:[0,1],automat:[0,1,5,18,20],avail:[2,6,9,10],avoid:20,axi:20,back:[0,14],backend:[4,5,6,11,18],base:[4,7,9,10,11,14,15],base_dataset:9,baseclass:[5,11],basedataset:[3,9,20],basemodel:[5,20],basic:[6,20],batch:[0,1,2,4,5,7,10,11,20],batch_data:[2,10],batch_siz:[0,1,20],becom:6,been:[2,4],befor:[0,2,4,10,18,20],before_train:[2,7,10],begin:20,behavior:[0,2,5,16],being:2,bellow:6,below:[18,20],best:[1,2,10,20],bestsaverhook:[10,20],better:[10,20],between:[2,7,14],big:[1,20],bit:20,blank:10,block:20,blur:[0,1],blur_prob:[0,1],bool:[7,10,11,20],both:[0,5],branch:18,broken:17,build:[0,4,18,20],built:2,burdensom:6,caffe2:20,calcul:20,call:[2,4,5,7,10,20],can:[0,1,2,4,5,6,18,19,20],cannot:5,carefulli:2,cast:20,cat:1,catch_sigint:10,catchsigint:[2,10],caught:10,certain:10,chang:[0,1,20],check:[1,7,10,19],checkpoint:5,checksum:1,child:10,class_nam:16,classif:[1,2],classifi:[0,1],classificationinfohook:[0,2],clear:0,cli:[1,13],clone:18,closer:2,cntk:20,code:[5,6,10,15,16,18,19],cognexa:[18,19],collect:[1,10],column:10,com:[18,19],combin:0,come:6,comfort:0,command:[1,4,5,6,14,15,18,20],comment:[0,1],common:0,complet:[0,1,20],compon:[0,3,5,6,19,20],comput:[0,2,5,10,20],compute_stat:10,computestat:[0,2,10],concept:[1,9],conclus:3,condit:10,config:[0,1,2,4,6,7,8,9,12,17,20],config_fil:[6,13],config_path:6,config_str:9,config_to_fil:13,config_to_str:13,configur:[1,3,5,8,9,10,11,13],consequ:20,consid:5,consist:[1,20],constant:[7,17,20],construct:[0,1,5,10,11,20],constructor:[0,1,2,4,5,6,9,11,16,20],consum:1,contain:[0,1,2,5,8,9,10,14,15,16,20],continu:5,contrari:20,contrib:20,conveni:0,convent:[1,10,20],convert:20,core:4,correct:20,correspond:[1,4,5,20],could:[1,16],counter:10,cover:[11,20],creat:[1,2,4,5,7,9,10,15,16,20],create_object:16,create_optim:20,creation:[2,20],criteria:10,crucial:0,csv:[2,10,20],ctrl:2,current:[0,10,11,20],custom:[0,20],cxf_config_fil:8,cxf_full_date_format:8,cxf_hooks_modul:8,cxf_log_date_format:8,cxf_log_fil:8,cxf_log_format:8,cxflow:[0,1,2,3,4,5,17,18],cxflow_scikit:[0,2],cxflow_tensorflow:[0,2,5,20],data:[0,2,3,4,7,10,20],data_root:0,databas:[0,1,20],dataset:[2,3,5,7,11,17],date:8,datset:0,debug:[6,20],decid:2,decod:[1,9,20],deep:[0,19],def:[1,2,20],default_valu:10,defaultdict:10,defin:[0,1,5,6,9,19,20],delet:5,delimit:10,demonstr:[1,5,20],denot:20,dens:20,depend:[4,18],depict:1,deploi:0,deriv:2,describ:[1,4,10,20],descript:20,design:[6,20],desir:20,detail:[2,3,4,20],detect:10,determin:[5,10],dev:[14,18],develop:[0,1,19],deviat:20,dict:[0,1,2,7,10,11,13,15,16],dictionari:[5,10,20],differ:[0,4,5,16],dig:[2,19],dim:20,dimens:20,dir:[9,11],directli:[0,10,18,20],directori:[0,2,5,6,8,9,10,11,13,20],disabl:14,disabledlogg:14,disabledprint:14,distinct:0,dive:[3,18],divis:10,document:[0,20],doe:[0,4,7,10,20],doesn:[0,1],dog:1,don:[0,1,20],done:[0,5,20],dot:[0,16],doubl:[7,10],down:17,download:1,drive:1,dtype:20,dump:[5,8,13],duplic:0,dure:[0,1,2,4,10,20],each:[0,2,4,5,10,20],easi:20,easier:20,easili:[0,1,15,19,20],either:[0,1,10,19],eleg:1,els:20,embed:0,emploi:[0,1,20],empti:[2,7,10],enabl:[0,1],encapsul:1,encod:[0,1,20],encourag:20,end:[4,15],enforc:0,enough:20,enter:[2,10],entir:14,entri:[0,2,5,9,10,15],enumer:10,environ:[5,20],environemt:5,epoch:[0,2,4,7,10,20],epoch_data:[2,10],epoch_id:[2,5,10],epoch_limit:[2,20],epoch_profil:2,epochstopperhook:[2,20],equal:20,error:[0,7,10,16,20],estim:20,etc:[1,4,5,6,10],eval_batch_train:10,evalu:[0,1,4,5,7,20],evaluate_stream:7,evaluet:0,even:[2,4],event:[3,4,7,10,15,20],everi:[2,5,9,10,20],everyth:[2,20],eviron:20,exactli:[0,10],exampl:[0,1,2,5,10,19,20],exce:10,exceed:10,except:16,execut:[1,4,10,15,18,20],exist:[5,11],exit:10,expect:[0,1,2,5,10,20],experi:[0,1],explain:[6,20],explicit:20,expos:11,extend:[10,13,20],extra:[1,4,20],extra_stream:[0,1,4,7,10,20],extrem:[1,20],fact:2,fail:[10,11],failur:10,fallback:11,fals:[5,7,11],far:[10,15],favourit:20,featur:[0,20],fed:[0,1,4],feed:11,feel:20,fetch:[1,20],few:[0,20],figur:1,file:[0,1,2,5,6,8,10,11,13,18,20],filenam:13,filesystem:5,finali:20,find:16,find_class_modul:16,fine:[0,4],finish:[2,10,20],first:[0,1,2,4,5,10,20],fit:14,fixed_batch_s:[0,7],float32:20,flow:10,focu:6,focus:20,follow:[0,1,2,3,4,5,6,16,17,18,20],form:[0,1,10,20],format:[2,8,10,13],former:1,forward:11,found:[4,10,16,20],four:[1,6,20],fq_name:16,fraction:20,framework:[2,20],free:20,from:[0,1,2,4,5,6,9,10,13,15,16,18,20],fscore:10,full:[8,10,16],fulli:[0,10,11,16],fundament:20,further:[0,1],furthermor:0,futur:[19,20],gener:[0,1,8,20],get:[2,7,16,20],get_class_modul:16,get_stream:7,git:18,github:[18,19],given:[0,1,2,5,7,9,10,11,13,16,20],global_variables_initi:20,goal:6,going:20,gold_vari:[0,2],good:[2,20],grace:10,graph:20,greater_equ:20,ground:0,handi:16,handl:[1,2],handler:10,happen:[4,20],has:[1,2,4,5,9,10,16,18],have:[0,2,7,9,10,16,19,20],header:10,height:[0,1],helper:15,henc:[0,1,2,20],here:[5,10,14,15],hidden:[0,1,20],hidden_activ:20,hook:[3,5,7,8,17],how:[2,10,20],howev:[0,1,2,5],http:[18,19],idea:20,ident:0,ids:0,ignor:[0,1,7,10,14],imag:[0,1,2,5],imagin:2,img1:1,img2:1,img3:1,img4:1,immedi:[0,10],implement:[1,2,3,4,5,9,11,20],imposs:0,includ:[0,1,10,20],incorrectli:0,increas:[6,10],index:10,indic:4,individu:[0,20],infer:[1,3,5,7,10],inference_logging_hook:0,inferencelogginghook:0,info:[2,10],inform:[0,3,4,5,6,11,20],inherit:[1,9,10],initi:[2,9,10],initil:20,input:[0,1,5,10,11,20],input_nam:[5,11],insid:2,instal:[6,20],instanc:[0,1,2,5,15,16],instanti:5,instead:[0,1,6,9,16,20],instruct:2,instrument:[2,6],integr:[3,20],intend:6,interfac:[0,5,10,11,20],internet:1,interrupt:[2,4,10],introduc:[0,20],invoc:10,invok:[0,1,2,6,20],ioerror:10,issu:[4,19],item:[4,20],iter:[1,4,7,10,11,13,16,20],its:[0,1,2,4,5,10,20],itself:[0,4,10],json:[0,20],just:[0,5,6],keep:[1,2],kei:[1,5,13,20],kera:20,keyerror:10,know:[0,1,5,19,20],kwarg:[0,1,2,9,10,11,16,20],label:[0,2],languag:0,last:2,later:[10,20],latter:1,layer:20,lazi:3,lazili:1,learn:[0,2,5,11,19,20],learning_r:[0,20],least:10,leav:1,len:20,length:20,less:0,let:[1,2,19,20],level:[0,6,10],levelnam:8,lifecycl:[3,10],like:[2,10,20],line:[0,6,10],link:0,list:[0,2,5,7,10,11,15,16,18,19,20],list_submodul:16,load:[0,1,5,7,13,20],load_config:13,load_training_batch:1,local_variables_initi:20,locat:[1,19],log:[0,2,5,6,8,9,10,14,16,20],log_dir:11,log_profil:[0,10],log_vari:10,logger:14,logginghook:20,logprofil:[0,2,10],logvari:[2,10],longer:0,look:[1,2,5,20],loop:[2,3,5,7,10],loss:[0,2,5,10,20],lot:20,luckili:2,machin:[0,5,11,20],mai:[0,1,2,5,9,10,13,20],mail:19,main:[1,2,3,7,10],main_loop:[0,1,2,4,7,20],mainloop:[5,7,9,10,11,20],maintain:10,major:[2,18,20],majority_dataset:20,majority_net:20,majoritydataset:20,majorityexampl:20,majorityexample_:20,majoritynet:20,make:[9,20],manag:[2,5,19],mandatori:0,map:[1,7,10,11],mappingproxi:16,match:0,matter:[2,4],max:10,max_epoch:10,maximum:10,mean:[0,2,10,20],meant:[10,20],measur:[0,10,15],median:10,memori:5,messag:8,method:[0,2,3,4,5,6,9,10,11,16,20],metric:[0,10],might:[0,1,4,18,20],mimic:16,min:10,minim:[10,20],minut:19,misc:[7,12,17],miss:[7,10],missing_variable_act:10,mit:19,mix:0,mlp:20,mode:7,model:[1,2,3,6,7,10,17,19],modifi:[2,20],modul:[0,1,2,7,8,9,10,11,13,14,15,16,20],modular:[0,20],module_nam:16,moment:[1,10,20],monitor:10,more:[0,2,3,4,20],moreov:16,most:[5,10,20],motiv:0,msec:8,multipl:[4,16],must:[0,9,18,20],my_data:0,my_dataset:[0,1],my_hook:0,my_logger_nam:14,my_model:0,my_modul:0,my_project:2,my_work:15,mydataset:[0,1],myhook:[0,2],mymodel:0,n_epoch:10,n_hidden_neuron:0,name:[0,1,2,4,5,6,7,8,10,11,13,14,15,16,18,20],name_suffix:[5,10,11],natur:[0,10,20],necessari:[1,20],need:[0,1,2,5,20],nest:[0,10],net:[4,20],network:[0,20],neural:0,neuron:0,never:4,nevertheless:[0,18],new_valu:10,newli:20,next:20,nightli:18,node:20,non:16,none:[2,7,9,10,11,14,16,20],nonetyp:[7,10,11],note:[0,1,2,5,20],noth:[2,10],notimplementederror:9,now:[0,2,20],npr:20,num:1,number:[0,2,4,10,20],numpi:[10,20],object:[0,1,2,5,7,9,10,11,14,15,16],obligatori:9,observ:[0,2,20],obtain:[0,9],offici:[10,19,20],often:[5,10],omit:20,on_failur:10,on_missing_vari:10,on_save_failur:10,on_unknown_typ:10,on_unused_sourc:[0,7],onc:[0,2,5,10,20],one:[1,2,4,5,7,9,10,20],ones:[0,2,20],onli:[0,1,2,4,5,10,20],oper:[1,2,20],opportun:10,optim:20,option:[0,7,9,10,11,16,18,20],order:[0,1,2,5,9,10,18,20],origin:[5,10,14],other:[0,1,10,14,20],otherwis:11,our:[1,2,5,19,20],out:[19,20],output:[0,2,4,5,6,8,9,10,11,13,20],output_dir:[2,9,10,13],output_fil:10,output_nam:[5,10,11],outsid:2,over:[2,20],overal:0,overlap:20,overrid:[2,9,13],overridden:9,own:[2,5,10],packag:[17,20],param:11,paramet:[0,1,5,7,9,10,11,13,15,16,20],pars:[1,9,13,16],parse_arg:13,parse_fully_qualified_nam:16,part:[0,5,20],pass:[0,2,4,5,6,9,11,14,15,16,20],path:[5,6,11,13],peek:2,perfectli:4,perform:[1,4,10,20],performac:20,persist:5,phase:[0,4],philosophi:3,pip:18,pipelin:1,place:2,placehold:20,pleas:[19,20],plot:1,plot_histogram:1,plugin:18,point:[0,2,5],polit:10,posibl:10,posit:5,possibl:[0,1,7,10,20],pow:20,practic:1,precis:20,predict:[0,1,2,3,5,7,20],predict_stream:[1,7,20],predicted_anim:5,predicted_vari:[0,2],predit:2,prescrib:9,present:[10,20],print:[1,10,14],print_statist:1,probabl:[0,1],proce:20,process:[0,2,3,4,5,10,11,20],produc:[0,10,20],producion:20,product:[0,5,20],profil:[2,4,7,10,12,17],programmat:6,progress:20,project:[19,20],proper:[1,6],properli:[10,18],properti:5,provid:[0,1,2,5,7,10,13,17,20],pseudocod:[4,5],publish:5,purpos:[1,18,20],put:6,python:[1,2,9,15,18,20],qualifi:[0,10,11,16],quallifi:16,queri:[1,4],quit:[2,10],rabbit:1,rais:[2,7,9,10,16],random:[0,1,20],random_integ:20,randomli:20,rang:[1,5,20],rate:0,ratio:20,read:10,read_data_train:10,readi:[0,20],real:20,realli:[1,20],reason:[0,1],receiv:7,recogn:10,recognit:5,recommend:[0,1,18,20],redirect:14,reduc:0,reduce_mean:20,refer:[1,10,19],reflect:[7,12,17],regard:4,regardless:[2,10],regist:[0,1,4,10,20],registr:[0,1,20],regular:[1,3,20],relat:[1,20],remain:0,remov:7,renam:20,repositori:18,repres:[1,2,20],requir:[0,1,9,10,20],required_min_valu:10,reset:10,reshap:20,resolut:0,respect:[2,10],respons:[4,20],rest:0,restor:[3,11,14],restore_fallback:[5,11],restore_from:[5,6,11],result:[0,2,10,11],resum:[1,5,20],resus:0,retriev:10,reus:0,reusabl:[19,20],rework:4,rewrit:0,rgb:1,right:4,rotat:[0,1],roughli:2,row:10,rtype:11,run:[3,4,7,10,11,18,20],run_predict:7,run_train:7,safe:20,sake:20,same:[0,16],save:[0,2,5,10,11,13,15,20],save_failure_act:10,save_model:10,savebest:[2,10],saveeveri:10,scalar:10,score:2,script:1,search:16,searchabl:16,second:[0,1,2,5,10,20],section:[0,1,3,4,11,19,20],see:[0,2,3,5,10,18,19,20],select:4,self:[1,2,7,10,20],self_num_sigint:10,sentenc:1,separ:[0,1,16,20],serv:0,set:[0,1,7,10],setup:18,sever:[1,17],shall:[2,5,20],shape:[1,20],share:[2,6,10],should:[1,4,5,6,9,10,11,15,20],sigint:[10,20],siginthook:20,signal:[10,20],signum:10,similar:[4,5],similarli:20,simpl:[0,1,2,4,15,20],simpler:20,simplest:18,simpli:20,simplic:[1,20],sinc:[0,20],singl:[1,4,5,10,20],situat:0,size:[0,1,4,7],skip:[0,7],skip_zeroth_epoch:[0,7],smart:[19,20],snippet:[0,5,20],some:[0,1,2,20],somebodi:5,sometim:1,sourc:[0,1,2,5,7,9,10,11,13,14,15,16,18,19,20],span:15,special:[0,2],specif:[0,4,5,6,11],specifi:[0,1,2,5,7,10,11,14,20],speed:19,spent:10,split:[1,20],sq_err:20,squar:20,standard:[0,2,8,10,20],start:[2,4,6,10,15],stat:10,statist:[1,2,5,10,20],statshook:20,std:[10,20],stderr:10,stdout:14,stem:10,step:[4,20],stop:[2,10,15,20],stop_aft:10,stopaft:[2,10],store:[2,5,6,10,11,20],str:[5,7,9,10,11,13,15,16],stream:[0,1,2,4,7,9,10,20],stream_nam:[2,7,9,10],string:[0,1,9,13,20],strongli:0,structur:[0,2],sub:[6,16],subclass:5,submodul:[11,16],subsequ:[2,10],subset:10,successfulli:5,suffix:[10,11,20],suggest:[0,1,4],suit:[0,18],sum:20,summar:10,superset:0,supplement:10,suppli:2,support:[0,10],suppos:[0,1,20],suppress:0,sure:20,syntax:0,system:2,t_co:16,take:[1,2,7,9,10],taken:[7,10,20],target:[0,13],task:1,tell:20,tensor:1,tensorboard:2,tensorboardhook:0,tensorflow:[5,18,20],termin:[0,2,4,10],test:[0,1,2,3,10,18,20],test_batch:5,test_i:20,test_stream:[1,5,20],test_x:20,than:[2,10],thank:10,thei:[0,1,20],them:[0,1,2,10,20],therefor:[0,4,20],thi:[0,1,2,3,4,5,9,10,11,14,16,18,20],thing:[4,20],third:20,those:[4,20],threshold:10,through:[1,4,6,7,10,20],time:[1,10,15],timer:15,todo:[1,4,5],togeth:[1,6,20],token:0,top:0,topic:20,total:20,track:2,tracker:19,train:[0,1,2,3,5,7,8,10,11,20],train_batch:5,train_by_stream:7,train_i:20,train_op:20,train_stream:[1,4,5,7,9,20],train_x:20,trainabl:11,trainig:2,trainingtermin:[2,10],translat:2,tri:11,trigger:[2,4,10],truth:0,tune:0,tupl:[13,16],tutori:[2,5,19],two:[0,1,2,4,5],txt:18,type:[0,1,7,10,11,13,15,16],typeerror:10,typic:1,unanot:0,unbias:20,under:[1,2,15,19],understand:20,union:[7,10,11],unit:[1,20],unknown:10,unknown_type_act:10,unsupport:10,unus:7,unused_source_act:7,updat:[1,4,5,11],upon:2,usag:[6,14,15],use:[0,1,2,4,11,18,20],used:[0,1,4,5,8,10,11,20],useful:[1,2,20],useless:1,user:[0,10,18],using:[0,6,18,20],usual:[2,4,5,6,10,11,20],util:[7,10,17,18],valid:[0,1,2,4,10,20],valid_stream:1,valu:[0,2,4,10,13,20],valueerror:[7,10],variabl:[0,2,5,10,20],variable_nam:10,variou:[0,1,8,20],vector:20,verb:10,verbos:6,veri:[0,4,20],verif:20,via:[1,10,19],visual:1,wai:[0,1,6,18],want:20,warn:[0,7,10,16],wast:1,watch:[2,20],webpag:0,well:[0,1,20],were:10,what:[0,2,4,5,20],when:[0,1,4,5,7,10,11,16,20],whenc:0,where:[2,4,5,20],wherein:10,whether:[1,4,5,10],which:[0,1,2,4,5,6,9,10,11,13,15,16,18,19,20],whole:[0,1,2,4,5,6,20],whose:1,width:[0,1],wise:1,wish:5,within:10,without:[1,5,10],work:2,workflow:5,world:20,would:[0,1,2,5,8,20],wrap:[16,20],wrapper:20,write:[2,10],write_csv:10,writecsv:[2,10],writetensorboard:2,written:[2,9,10],wrong:10,xxx:10,y_hat:20,yaml:[0,1,8,9,13,20],yet:10,yield:[1,20],you:[0,2,5,19,20],your:[2,4,19,20],zero:20,zeroth:7},titles:["Configuration","Dataset","Hooks","Advanced","Main Loop","Model","CLI Reference","<code class=\"docutils literal\"><span class=\"pre\">cxflow</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow.constants</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow.datasets</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow.hooks</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow.models</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow.utils</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow.utils.config</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow.utils.misc</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow.utils.profile</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow.utils.reflection</span></code>","API Reference","Getting Started","cxflow","Tutorial"],titleterms:{"class":[7,9,10,11,14,15],"function":[13,16],The:1,Using:20,addit:1,advanc:3,after_batch:2,after_epoch:2,api:17,argument:6,basedataset:1,cli:6,conclus:0,config:13,configur:[0,2,20],constant:8,contribut:19,cxflow:[6,7,8,9,10,11,12,13,14,15,16,19,20],data:1,dataset:[0,1,4,6,9,20],develop:18,event:2,except:10,get:18,hook:[0,2,4,10,20],infer:0,instal:18,integr:4,introduct:20,lazi:1,licens:19,lifecycl:4,loop:[0,4,20],main:[0,4,20],method:1,misc:14,model:[0,4,5,11,20],philosophi:1,posit:6,predict:[4,6],process:1,profil:15,refer:[6,17],reflect:16,regular:2,requir:18,restor:5,resum:6,run:5,start:18,submodul:[7,12],support:19,task:20,test:4,train:[4,6],tutori:20,util:[12,13,14,15,16],variabl:8}})