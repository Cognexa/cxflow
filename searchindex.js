Search.setIndex({docnames:["advanced/config","advanced/dataset","advanced/hook","advanced/index","advanced/main_loop","advanced/model","cli","cxflow/cxflow","cxflow/cxflow.constants","cxflow/cxflow.datasets","cxflow/cxflow.hooks","cxflow/cxflow.models","cxflow/index","getting_started","index","tutorial"],envversion:55,filenames:["advanced/config.rst","advanced/dataset.rst","advanced/hook.rst","advanced/index.rst","advanced/main_loop.rst","advanced/model.rst","cli.rst","cxflow/cxflow.rst","cxflow/cxflow.constants.rst","cxflow/cxflow.datasets.rst","cxflow/cxflow.hooks.rst","cxflow/cxflow.models.rst","cxflow/index.rst","getting_started.rst","index.rst","tutorial.rst"],objects:{"":{cxflow:[7,0,0,"-"]},"cxflow.MainLoop":{EMPTY_ACTIONS:[7,2,1,""],UNUSED_SOURCE_ACTIONS:[7,2,1,""],__init__:[7,3,1,""],_check_sources:[7,3,1,""],_create_epoch_data:[7,3,1,""],_run_epoch:[7,3,1,""],_run_zeroth_epoch:[7,3,1,""],_try_run:[7,3,1,""],epochs_done:[7,2,1,""],evaluate_stream:[7,3,1,""],extra_streams:[7,2,1,""],fixed_epoch_size:[7,2,1,""],get_stream:[7,3,1,""],run_evaluation:[7,3,1,""],run_training:[7,3,1,""],train_by_stream:[7,3,1,""]},"cxflow.constants":{CXF_BUFFER_SLEEP:[8,4,1,""],CXF_CONFIG_FILE:[8,4,1,""],CXF_DEFAULT_LOG_DIR:[8,4,1,""],CXF_FULL_DATE_FORMAT:[8,4,1,""],CXF_HOOKS_MODULE:[8,4,1,""],CXF_LOG_DATE_FORMAT:[8,4,1,""],CXF_LOG_FILE:[8,4,1,""],CXF_LOG_FORMAT:[8,4,1,""],CXF_NA_STR:[8,4,1,""],CXF_PREDICT_STREAM:[8,4,1,""],CXF_TRACE_FILE:[8,4,1,""]},"cxflow.datasets":{AbstractDataset:[9,1,1,""],BaseDataset:[9,1,1,""],DownloadableDataset:[9,1,1,""],StreamWrapper:[9,1,1,""]},"cxflow.datasets.AbstractDataset":{__init__:[9,3,1,""]},"cxflow.datasets.BaseDataset":{__init__:[9,3,1,""],_configure_dataset:[9,3,1,""],stream_info:[9,3,1,""]},"cxflow.datasets.DownloadableDataset":{_configure_dataset:[9,3,1,""],data_root:[9,2,1,""],download:[9,3,1,""],download_urls:[9,2,1,""]},"cxflow.datasets.StreamWrapper":{__enter__:[9,3,1,""],__exit__:[9,3,1,""],__init__:[9,3,1,""],__iter__:[9,3,1,""],__next__:[9,3,1,""],_dequeue_batch:[9,3,1,""],_enqueue_batches:[9,3,1,""],_epoch_limit_reached:[9,3,1,""],_get_stream:[9,3,1,""],_next_batch:[9,3,1,""],_start_thread:[9,3,1,""],_stop_thread:[9,3,1,""],allow_buffering:[9,2,1,""],empty:[9,3,1,""],name:[9,2,1,""]},"cxflow.hooks":{AbstractHook:[10,1,1,""],AccumulateVariables:[10,1,1,""],Check:[10,1,1,""],ComputeStats:[10,1,1,""],EveryNEpoch:[10,1,1,""],LogDir:[10,1,1,""],LogProfile:[10,1,1,""],LogVariables:[10,1,1,""],OnPlateau:[10,1,1,""],SaveBest:[10,1,1,""],SaveConfusionMatrix:[10,1,1,""],SaveEvery:[10,1,1,""],SaveLatest:[10,1,1,""],ShowProgress:[10,1,1,""],StopAfter:[10,1,1,""],StopOnNaN:[10,1,1,""],StopOnPlateau:[10,1,1,""],TrainingTerminated:[10,6,1,""],WriteCSV:[10,1,1,""]},"cxflow.hooks.AbstractHook":{__init__:[10,3,1,""],after_batch:[10,3,1,""],after_epoch:[10,3,1,""],after_epoch_profile:[10,3,1,""],after_training:[10,3,1,""],before_training:[10,3,1,""]},"cxflow.hooks.AccumulateVariables":{__init__:[10,3,1,""],_reset_accumulator:[10,3,1,""],after_batch:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.hooks.Check":{__init__:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.hooks.ComputeStats":{EXTRA_AGGREGATIONS:[10,2,1,""],__init__:[10,3,1,""],_compute_aggregation:[10,5,1,""],_raise_check_aggregation:[10,5,1,""],_save_stats:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.hooks.EveryNEpoch":{__init__:[10,3,1,""],_after_n_epoch:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.hooks.LogDir":{__init__:[10,3,1,""],after_epoch:[10,3,1,""],after_training:[10,3,1,""],before_training:[10,3,1,""]},"cxflow.hooks.LogProfile":{after_epoch_profile:[10,3,1,""]},"cxflow.hooks.LogVariables":{UNKNOWN_TYPE_ACTIONS:[10,2,1,""],__init__:[10,3,1,""],_log_variables:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.hooks.OnPlateau":{OBJECTIVES:[10,2,1,""],_AGGREGATION:[10,2,1,""],__init__:[10,3,1,""],_on_plateau_action:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.hooks.SaveBest":{OBJECTIVES:[10,2,1,""],__init__:[10,3,1,""],_get_value:[10,3,1,""],_is_value_better:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.hooks.SaveConfusionMatrix":{FIGURE_ACTIONS:[10,2,1,""],__init__:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.hooks.SaveEvery":{SAVE_FAILURE_ACTIONS:[10,2,1,""],__init__:[10,3,1,""],_after_n_epoch:[10,3,1,""],save_model:[10,5,1,""]},"cxflow.hooks.SaveLatest":{__init__:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.hooks.ShowProgress":{__init__:[10,3,1,""],after_batch:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.hooks.StopAfter":{__init__:[10,3,1,""],_check_train_time:[10,3,1,""],after_batch:[10,3,1,""],after_epoch:[10,3,1,""],before_training:[10,3,1,""]},"cxflow.hooks.StopOnNaN":{UNKNOWN_TYPE_ACTIONS:[10,2,1,""],__init__:[10,3,1,""],_check_nan:[10,3,1,""],_is_nan:[10,3,1,""],after_batch:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.hooks.StopOnPlateau":{_on_plateau_action:[10,3,1,""]},"cxflow.hooks.WriteCSV":{MISSING_VARIABLE_ACTIONS:[10,2,1,""],UNKNOWN_TYPE_ACTIONS:[10,2,1,""],__init__:[10,3,1,""],_write_header:[10,3,1,""],_write_row:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.models":{AbstractModel:[11,1,1,""],Ensemble:[11,1,1,""],Sequence:[11,1,1,""]},"cxflow.models.AbstractModel":{__init__:[11,3,1,""],input_names:[11,2,1,""],output_names:[11,2,1,""],restore_fallback:[11,2,1,""],run:[11,3,1,""],save:[11,3,1,""]},"cxflow.models.Ensemble":{AGGREGATION_METHODS:[11,2,1,""],__init__:[11,3,1,""],_load_models:[11,3,1,""],input_names:[11,2,1,""],output_names:[11,2,1,""],restore_fallback:[11,2,1,""],run:[11,3,1,""],save:[11,3,1,""]},"cxflow.models.Sequence":{__init__:[11,3,1,""],_load_models:[11,3,1,""],input_names:[11,2,1,""],output_names:[11,2,1,""],restore_fallback:[11,2,1,""],run:[11,3,1,""],save:[11,3,1,""]},cxflow:{MainLoop:[7,1,1,""],constants:[8,0,0,"-"],datasets:[9,0,0,"-"],hooks:[10,0,0,"-"],models:[11,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","data","Python data"],"5":["py","staticmethod","Python static method"],"6":["py","exception","Python exception"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:data","5":"py:staticmethod","6":"py:exception"},terms:{"03d":8,"0th":[0,7],"10th":10,"12s":8,"1st":[2,9,10],"2nd":[2,9],"abstract":[10,11],"boolean":[4,5],"case":[0,1,2,5,7,9,10,15],"class":[0,1,2,5,15],"default":[4,6,8,9,10],"final":[1,4,13,15],"float":[9,10],"function":[1,4,7,9,10,15],"import":[1,2,5,8,11,15],"int":[0,1,2,7,9,10,15],"long":[6,10],"new":[0,1,2,5,9,10,11,15],"return":[1,4,5,7,9,10,11,15],"short":[2,10],"static":10,"super":2,"true":[0,1,5,7,9,10,11],"try":[7,15],"var":[0,11],"while":[4,5],For:[0,1,2,5,10,15],One:[4,10],That:1,The:[0,2,4,5,6,8,9,10,11,13,14,15],Then:[9,15],There:4,These:[0,1,5],With:[0,1,2,6,15],__enter__:9,__exit__:9,__init__:[0,2,7,9,10,11],__iter__:9,__next__:9,_accumul:10,_after_n_epoch:10,_aggreg:10,_check_nan:10,_check_sourc:7,_check_train_tim:10,_compute_aggreg:10,_configure_dataset:[1,9,15],_create_epoch_data:7,_create_model:15,_dataset:[10,15],_dequeue_batch:9,_enqueue_batch:9,_epoch:10,_epoch_limit:2,_epoch_limit_reach:9,_get_stream:9,_get_valu:10,_is_nan:10,_is_value_bett:10,_iter:10,_load_model:11,_log_vari:10,_minut:10,_model:11,_next_batch:9,_on_missing_vari:10,_on_plateau_act:10,_on_unknown_typ:10,_on_unused_sourc:7,_raise_check_aggreg:10,_reset_accumul:10,_run_epoch:7,_run_train:7,_run_zeroth_epoch:7,_save_stat:10,_start_thread:9,_stop_thread:9,_stream:[1,9,10,15],_test_i:15,_test_x:15,_train_i:15,_train_x:15,_try_run:7,_variabl:10,_write_head:10,_write_row:10,abil:1,abl:[5,10,15],about:[5,10],abov:[2,10,15],absolut:10,abstract_hook:2,abstractdataset:[0,1,7,9,10,11,15],abstracthook:[0,2,7,10],abstractmodel:[0,4,5,7,9,10,11],accept:[2,5,15],access:6,accord:[11,15],accumul:[2,10,11],accumulate_vari:10,accumulatevari:[2,10],accur:11,accuraci:[0,10,15],act:2,action:[0,2,7,10,15],actual:[0,1,2,6],adamoptim:15,add:15,adding:15,addit:[0,2,7,9,10,11,13,15],addition:[5,9,10],advanc:[0,9,14,15],after:[2,4,9,10,15],after_batch:[7,10],after_batch_hooks_train:10,after_epoch:[7,10],after_epoch_hook:10,after_epoch_profil:[2,7,10],after_train:[2,7,10],again:[0,15],aggreag:10,aggreg:[10,11],aggregation_method:11,agnost:15,alik:5,all:[0,1,2,4,5,6,7,9,10,11,15],allow:[1,2,9,10,15],allow_buff:9,along:[1,4,15],alongsid:1,alreadi:[2,9,15],also:[0,2,4,5,11,15],altern:[4,9],although:[5,6],altogeth:2,alwai:[4,9],among:10,analog:1,anchor:0,ani:[0,1,2,4,5,9,10,11,15],anim:[0,1,5],annoi:1,annot:10,anoth:0,answer:15,anyth:15,anywai:10,api:[6,9,10,14,15],append:11,appli:11,applic:[6,11],approach:0,arbitrari:[1,6,10,15],arbitrarili:0,aren:0,arg1:2,arg2:2,arg:[2,9,11],argument:[0,1,2,4,5,10,15],artifact:15,asctim:8,assembl:11,assertionerror:[0,7,10,11],attempt:5,attribut:[0,11,15],attributeerror:7,augment:[0,1],automat:[0,1,5,13,15],avail:[0,2,6,9,10],avoid:15,axi:15,back:[0,15],backend:[4,5,6,11,13],bar:10,base:[4,5,7,9,10,11],baseclass:11,basedataset:[9,10,15],basemodel:[5,15],basic:[6,15],batch:[0,1,2,4,5,7,8,9,10,11,15],batch_data:[2,10],batch_siz:[0,1,15],batchdata:1,beauti:1,becom:6,been:[2,4],befor:[2,4,8,10,11,13,15],before_train:[2,7,10],begin:15,behavior:[2,5],being:[0,2],below:[5,6,13],best:[1,2,10,15],better:10,between:[0,2,7],bia:15,big:[1,15],bit:15,blank:10,block:[9,15],blue:10,blur:[0,1],blur_prob:[0,1],bool:[0,7,9,10,11,15],both:[5,10],branch:13,buffer:[0,7,8,9,11],buffer_s:9,build:[0,13,15],built:2,bundl:1,burdensom:6,caffe2:15,calcul:15,call:[1,2,4,5,7,10,11,15],call_native_backend:9,callabl:[7,9],can:[0,1,2,4,5,6,9,10,11,13,15],cannot:5,caption:10,carefulli:2,cast:15,cat:1,catchsigint:2,caughtinterrupt:7,certain:[10,15],chang:[0,15],check:[1,7,9,10,14],checkpoint:5,checksum:1,child:10,childprocesserror:9,class_with_index_on:10,class_with_index_thre:10,class_with_index_zero:10,classes_nam:10,classes_names_method_nam:10,classifi:[0,1],classificationinfohook:0,clear:0,cli:9,clone:13,closer:2,cmap:10,cntk:15,code:[5,6,13,14],cognexa:[13,14],collect:10,color:10,colorbar:10,colormaps_refer:10,column:10,com:[13,14],combin:0,come:[6,10],comfort:0,command:[1,4,5,6,9,13,15],comment:[0,1],common:0,compat:1,complet:[1,6,15],complex:1,compli:1,compon:[0,1,3,5,6,15],comput:[0,2,5,10,15],compute_stat:10,computestat:[0,2,10,15],concept:[1,9],condit:10,config:[1,2,4,6,8,9,10,11,15],config_fil:6,config_path:6,config_str:9,configur:[1,3,5,8,9,10,11],confus:10,consequ:[4,15],consid:[5,10],consist:15,constant:[7,10,12,15],construct:[0,1,5,9,10,11,15],constructor:[0,1,2,4,5,6,9,11,15],consum:[1,9],contain:[0,1,2,5,8,9,10,15],content:9,continu:[4,5],contrib:15,control:15,conveni:15,convent:10,convert:15,core:[4,7],correct:15,correctli:10,correspond:[4,5,9,15],could:1,count:10,counter:10,cover:11,cpu:9,creat:[0,1,2,4,5,7,9,10,11,15],creation:[2,15],criteria:10,crucial:0,csv:[2,10,15],ctrl:2,current:[10,11,15],custom:0,cut:[0,7,9],cxf_buffer_sleep:8,cxf_config_fil:8,cxf_default_log_dir:8,cxf_full_date_format:8,cxf_hooks_modul:8,cxf_log_date_format:8,cxf_log_fil:8,cxf_log_format:8,cxf_na_str:8,cxf_predict_stream:8,cxf_trace_fil:8,cxflow:[0,1,2,3,4,5,12,13],cxflow_scikit:0,cxflow_tensorflow:[0,2,5,15],cxtf:15,data:[0,2,4,7,9,10,15],data_root:[0,9],databas:[0,1,15],dataset:[2,3,5,7,10,11,12],date:8,debug:[6,15],decid:2,decod:[1,9,15],deep:0,def:[1,2,9,15],default_valu:10,defaultdict:10,defin:[0,1,5,6,9,10,15],delet:[5,6],delimit:10,demand:1,demonstr:[5,15],denot:15,dens:15,depend:[4,13],deriv:2,describ:[1,4,10,15],descript:15,design:6,desir:15,detail:[2,3,4,15],detect:10,determin:[5,10],dev:13,develop:0,dict:[1,2,7,10,11],dictionari:[0,1,5,10,15],differ:[0,1,5,7,11,15],dig:[2,14],dim:15,dimens:15,dir:[6,9,10,11],direct:9,directli:[0,9,10,13],directori:[0,2,5,6,8,9,10,11,15],disabl:9,displai:10,distinct:0,dive:[3,13],divid:[0,12],document:[0,15],doe:[0,7,10,11,15],doesn:1,dog:1,don:[0,15],done:[0,5,6,7,15],dot:14,doubl:[7,10],download:[1,9],download_filenam:9,download_url:9,downloadabledataset:9,drive:1,dtype:[9,15],dump:[5,8],durat:8,dure:[2,4,9,10],each:[0,2,4,5,10,15],eager_load:11,eas:1,easi:15,easier:15,easili:[0,1,15],either:[10,14],els:15,embed:0,emploi:15,empti:[0,2,7,9,10],empty_act:[0,7],enabl:[0,1],encod:[0,1,15],encourag:15,end:[4,9,15],enough:15,enqueu:9,ensembl:11,enter:[2,10],entri:[0,2,5,9,10],enumer:10,environ:[5,9,15],environemt:5,epoch:[0,2,4,6,7,9,10,15],epoch_data:[2,10],epoch_id:[2,5,10],epoch_limit:2,epoch_profil:2,epoch_s:9,epochs_don:7,epochstopperhook:2,equal:[10,15],error:[0,7,9,10,15],essenti:1,estim:15,eta:10,etc:[0,1,4,5,6,9,10,15],eval:[0,1,5,15],eval_batch_train:10,evalu:[1,5,7,15],evaluate_stream:7,even:[2,4],event:[4,7,9,10,15],everi:[1,2,5,9,10,15],every_n_epoch:10,everynepoch:10,everyth:[2,15],exactli:10,exampl:[0,1,2,5,9,10,15],exce:10,exceed:10,execut:[4,10,13,15],exist:[5,11,15],expect:[0,2,5,10,15],experi:[0,1,14,15],explain:[6,15],expos:11,extend:[10,15],extra:[1,4,7,9,10],extra_aggreg:10,extra_stream:[0,1,4,7,10,15],extract:9,extrem:[1,15],facilit:11,fact:[1,2],fail:[10,11],failur:10,fallback:11,fals:[0,5,6,7,9,10,11],far:10,favorit:15,featur:[0,9,15],fed:[1,4,11],feed:11,feel:15,fetch:[1,10,15],few:[0,15],field:1,figsiz:10,figur:[1,10],figure_act:10,file:[0,1,2,5,6,8,9,10,11,13,15],filenam:8,filesystem:5,find:[1,10,15],fine:4,finish:[2,6,10,15],first:[0,1,2,4,5,10,15],fix:[7,9],fixed_batch_s:[0,7],fixed_epoch_s:[0,7],flag:11,float32:15,flow:10,focu:[6,14],focus:15,follow:[0,1,2,3,4,5,6,12,13,15],form:[0,10,15],format:[1,2,6,8,10],forward:[0,11],found:[4,10,15],four:[6,15],fraction:15,framework:[2,14,15],free:15,from:[0,1,2,4,5,6,9,10,11,13,15],fscore:10,full:[8,10,15],fulli:[0,1,9,10,11],fundament:15,further:[0,1],furthermor:0,gener:[0,8,15],get:[2,7,9,10],get_stream:7,gil:9,git:13,github:[0,13,14,15],give:1,given:[0,1,2,5,6,7,9,10,11,15],goal:6,going:15,gold_vari:0,good:[2,9,15],graph:15,greater:10,greater_equ:15,green:10,ground:0,handl:[2,10],happen:4,has:[1,2,4,5,9,10,13,15],have:[0,2,6,7,9,10,14,15],header:10,height:0,henc:[0,2],here:[4,5,10],hidden:[0,1,15],hold:9,hook:[3,5,7,8,12],hour:10,how:[2,10,15],howev:[0,1,2,5,15],html:10,http:[10,13],ids:0,ignor:[1,7,10],imag:[0,1,2,5,11],imagin:[1,2],img1:1,img2:1,img3:1,img4:1,implement:[1,2,3,4,5,9,11,15],imposs:0,improv:10,includ:[0,6,10,15],increas:[6,10],index:10,indic:4,individu:[0,15],infer:[0,5,7,10,11],inference_logging_hook:0,inferencelogginghook:0,infin:10,info:[2,10,15],inform:[0,1,3,4,5,6,11,15],inherit:[1,9,10],initi:[1,2,10],input:[0,5,10,11,15],input_nam:[5,11],insid:2,instal:[6,15],instanc:[0,1,2,5],instanti:5,instead:[0,1,6,10,15],instruct:[2,9],instrument:[2,6,15],intact:4,integr:15,intend:[6,9],interfac:[0,5,10,11,15],internet:1,interoper:1,interrupt:[2,4,10],invalid:10,invoc:10,invok:[0,1,2,6,15],ioerror:10,issu:14,item:[4,15],iter:[0,1,4,7,9,10,11,15],its:[0,1,2,4,5,9,10,15],itself:[1,4,5,10],json:[0,15],just:[0,5,6,10,15],keep:[1,2,6,9],kei:[1,2,5,15],kera:15,keyerror:10,know:[0,1,5,14,15],kwarg:[1,2,9,10,11,15],label:[0,1,2,10],labels_nam:10,languag:[0,1],larg:1,last:[2,7,10],later:[10,15],latest:10,layer:15,learn:[0,1,2,5,11,14,15],learning_r:[0,15],least:[6,10],leav:[1,4],len:15,length:[7,15],let:[2,14,15],level:[1,6,10],levelnam:8,lifecycl:10,lightweight:[14,15],like:[0,2,15],limit:9,line:[0,6,10],link:0,list:[0,1,2,5,6,7,9,10,11,13,15],load:[0,1,5,7,11,15],load_training_batch:1,log:[0,1,2,5,6,8,9,10,15],log_dir:[10,11],log_profil:[0,10],log_vari:10,logdir:10,logprofil:[0,2,10],logvari:[2,10,15],long_term:10,look:[1,2,5,15],loop:[2,3,5,7,10],loss:[0,2,5,10,15],lot:15,lower:10,luckili:2,machin:[0,1,5,11,14,15],mai:[0,1,2,4,5,9,10,15],mail:14,main:[1,2,3,7,9,10],main_loop:[0,1,2,4,7,15],mainloop:[0,5,7,9,10,11,15],maintain:10,major:[2,13,15],major_vot:11,majority_dataset:15,majority_net:15,majoritydataset:15,majorityexampl:15,majorityexample_:15,majoritynet:15,make:[1,9,15],manag:[1,2,5,9,11,15],mandatori:0,mani:6,map:[7,9,10,11,15],mask:10,mask_nam:10,match:[0,15],matplotlib:10,matrix:10,matter:[1,2,4],max:10,max_epoch:10,maximum:10,mayb:[9,11],mean:[0,2,7,9,10,11,15],meant:[10,15],measur:[0,10],median:10,memori:5,messag:8,met:10,method:[0,2,4,5,6,9,10,11,15],metric:[0,10],might:[0,1,4,9,13,15],min:10,mini:1,minim:[10,15],minut:[10,14],misc:7,miss:[7,10],missing_variable_act:10,mit:14,mlp:15,mode:7,model:[1,2,3,6,7,9,10,12],model_path:11,models_path:11,models_root:11,modifi:[2,15],modul:[0,2,7,8,9,10,11,15],modular:[0,14,15],moment:[1,10,15],monitor:10,more:[0,1,2,3,4,6,10,11,15],most:[0,1,5,7,10,15],motiv:0,msec:8,multipl:[4,11],must:[1,9,10,13,15],my_data:0,my_dataset:[0,1],my_hook:0,my_model:0,my_project:2,my_stream:0,mydataset:[0,1],myensembl:11,myhook:[0,2],mymodel:0,mypipelin:11,n_epoch:10,n_exampl:15,n_hidden_neuron:0,name:[0,1,2,5,7,8,9,10,11,13,15],name_suffix:[5,10,11],nan:10,nancount:10,nanfract:10,nanmean:10,natur:[0,10,15],necessari:[1,15],need:[0,1,2,5,10,15],neither:11,nest:10,net:15,network:[0,15],neural:0,neuron:0,nevertheless:13,new_valu:10,newli:15,next:[1,8,9,15],nightli:13,node:15,non:[0,9],none:[0,2,7,9,10,11,15],normal:10,note:[0,2,5,15],noth:[2,10],notimplementederror:[9,11],now:[1,2,15],npr:15,num:1,num_batch:10,num_class:10,num_classes_method_nam:10,number:[0,2,4,7,9,10,15],numpi:[10,15],object:[0,1,2,5,7,9,10,11],obligatori:9,observ:[2,10,15],obtain:[0,9],offici:[10,14,15],often:[1,5,10],omnipres:0,on_empty_batch:[0,7],on_empty_stream:[0,7],on_failur:10,on_missing_vari:10,on_plateau:10,on_save_failur:10,on_unknown_typ:10,on_unkown_typ:10,on_unused_sourc:[0,7],onc:[0,2,5,10,15],one:[0,1,4,5,7,9,10,11,15],ones:[2,15],onli:[0,1,2,4,5,6,9,10,11,15],onplateau:10,oper:[1,2,15],opportun:10,optim:15,option:[0,7,9,10,11,13,15],order:[1,2,5,9,10,11,13],org:10,origin:[0,5,11],other:[1,10,15],otherwis:[10,11],our:[0,1,2,5,14,15],out:14,output:[0,2,4,5,6,8,9,10,11,15],output_dir:[2,9,10],output_fil:10,output_nam:[5,11],outsid:2,over:[2,15],overal:0,overlap:15,overrid:[0,2,9,10],overridden:[0,9,10],own:[2,5,10],packag:[12,15],paramet:[0,1,5,7,9,10,11,15],parametr:0,pars:[1,9],part:[0,5,15],particular:[1,15],pass:[0,1,2,4,5,6,9,10,11,15],path:[0,5,6,9,11],peek:2,perfectli:4,perform:[1,4,10,15],performac:15,period:10,persist:5,phase:[0,4],pip:13,pipelin:1,place:[2,9,10],placehold:15,plateau:10,pleas:[14,15],plot:[1,10],plot_histogram:1,plugin:13,point:[1,2,5],portion:1,posibl:10,posit:5,possibl:[7,9,10,11],pow:15,practic:1,preceed:11,precis:[11,15],predict:[0,2,7,8,10,11,15],predict_stream:0,predicted_anim:5,predicted_vari:0,predictions_nam:10,predit:2,prepar:9,preprocess_batch_in_python:9,prescrib:9,present:[10,15],pretti:8,principl:15,print:[1,6,8,10],print_statist:1,prior:4,probabl:[0,1],problem:1,proce:15,procedur:9,process:[0,2,4,5,8,10,11,15],produc:[0,9,10,15],product:[5,15],profil:[2,4,9,10],programmat:6,progress:[10,15],project:[11,14,15],proper:6,properli:[9,10,13,15],properti:[5,9,10],provid:[0,1,2,5,7,10,11,15],pseudocod:[4,5],publish:5,purpos:[1,13,15],put:6,python:[1,2,9,11,13,15],qualifi:[0,1,10,11],queri:[1,4],queue:9,quit:[1,2,15],rabbit:1,rais:[0,2,7,9,10,11],random:[0,1,15],random_integ:15,randomli:15,rang:[1,5,15],rapid:[14,15],rate:0,ratio:15,raw:9,reach:[9,10],read:[10,15],read_data_train:10,readi:15,real:15,reason:[0,1],receiv:[4,7],recogn:10,recognit:5,recommend:[1,13,15],record:[9,10],recurs:[6,10],refer:[0,10,14,15],referenc:0,regardless:[1,2,10],regist:[0,1,4,15],regular:15,rel:10,relat:[1,15],releas:9,releasedsemaphor:9,remain:0,remov:[0,7],renam:15,report:9,repositori:[0,13,15],repres:[2,10,15],requir:[0,1,9,10,15],required_min_valu:10,reset:[9,10],reshap:15,resolut:0,resourc:9,respect:[1,2,7,10,15],respons:[4,15],restor:[4,11],restore_fallback:[5,11],restore_from:[5,6,11],result:[0,2,10,11],resum:[5,15],resus:0,retriev:10,reusabl:15,right:4,risen:9,root:[0,9,11],rotat:[0,1],roughli:2,routin:9,row:10,run:[4,6,7,9,10,11,13,15],run_evalu:7,run_func:7,run_train:7,same:10,save:[0,2,5,9,10,11,15],save_cm:10,save_failure_act:10,save_model:10,savebest:[2,10,15],saveconfusionmatrix:10,saveeveri:10,savelatest:10,scalar:10,script:1,seamlessli:1,search:10,second:[2,5,15],section:[0,1,3,4,10,11,14,15],see:[0,1,2,3,5,10,13,14,15],select:4,self:[1,2,7,9,10,11,15],separ:[0,1,15],sequenc:[9,10,11],sequenti:11,set:[0,1,7,9,10,11],setup:13,sever:[1,12],shall:[5,15],shape:[9,15],share:[2,6,10],short_term:10,should:[1,4,5,6,9,10,11,15],show:10,show_progress:10,showprogress:10,signal:9,similar:[1,15],similarli:15,simpl:[0,1,2,4,11,15],simplest:13,simpli:[0,15],simultan:9,singl:[4,5,6,9,10,15],situat:0,size:[0,1,4,7,9,10],skeleton:1,skip:[0,7],skip_zeroth_epoch:[0,7],sleep:8,slightli:15,small:1,smaller:10,snippet:5,solv:15,some:[0,1,2,9,10,15],somebodi:5,sometim:[0,1],soon:10,sourc:[0,1,2,5,7,9,10,11,13,14,15],spare:9,special:[0,2],specif:[4,5,6,11],specifi:[0,1,2,4,5,7,9,10,11,15],spent:10,split:[1,15],squar:15,standard:[0,2,8,10],start:[2,4,6,8,9,10],stat:10,state:9,statist:[1,5,10],stderr:10,stem:10,step1:11,step2:11,step3:11,step:[4,11,15],stop:[2,6,9,10,15],stop_aft:10,stop_ev:9,stop_on_inf:10,stop_on_nan:10,stop_on_plateau:10,stopaft:[2,10,15],stopiter:9,stoponnan:10,stoponplateau:10,store:[2,5,6,10,11,15],str:[0,5,7,9,10,11],stream:[0,2,4,7,8,9,10,11,15],stream_fn:9,stream_info:9,stream_nam:[1,2,5,7,9,10],streamwrapp:[7,9,11],string:[0,1,8,9,10,15],structur:[0,2],stuff:9,sub:[0,6,11],subclass:5,subdir:6,subdirectori:6,submodul:11,subsequ:2,subset:10,successfulli:5,suffix:[10,11,15],suggest:1,suit:[0,13],sum:15,summar:10,superset:0,supplement:10,suppli:2,support:[0,10,11],system:2,t_co:9,take:[0,2,7,9,10],taken:[7,10,11,15],target:[0,15],techniqu:1,tell:15,tensorboard:2,tensorboardhook:0,tensorflow:[5,13,15],termin:[0,2,4,9,10],test:[0,1,2,4,10,13,15],test_batch:5,test_stream:[1,5,15],than:[2,10],thank:10,thei:[0,1,15],them:[1,2,9,10,11,15],therefor:[0,4,15],thi:[0,1,2,3,4,5,6,9,10,11,13,15],thing:15,those:[4,15],though:1,thread:9,threshold:10,through:[1,4,6,7,10,15],thu:15,time:[1,9,10],togeth:[0,1,6,11],token:0,total:[10,15],total_batch_count:10,trace:[7,8],track:2,tracker:14,train:[0,1,2,5,7,8,9,10,11,15],train_batch:5,train_by_stream:7,train_stream:[1,4,5,9,10,15],train_stream_nam:[0,7,10],trainabl:11,trainig:2,trainingtermin:[2,10],translat:[2,10],tri:11,trigger:[2,4,10,15],truli:1,truth:0,tupl:[1,10],tutori:[2,5,14],two:[1,2,4,5,7],txt:13,type:[0,7,9,10,11],typeerror:10,typic:[1,9,11,15],ultim:11,unannot:0,undefin:10,under:[1,2,11,14],underli:[9,11],understand:15,unexpectedli:9,unit:15,unknown:10,unknown_type_act:10,unsupport:[0,7,10],unsurprisingli:0,unus:[0,7,11],unused_source_act:[0,7],updat:[1,4,5,11],upon:2,url:9,url_root:9,usabl:[14,15],usag:[1,6,9,11],use:[0,1,2,4,6,9,11,13,15],used:[0,1,5,8,9,10,11,15],useful:[1,2,11,15],useless:1,user:[10,13],uses:[0,9],using:[6,11,13,15],usual:[2,4,5,6,10,11,15],util:[7,9,10,13],valid:[0,1,2,4,10,15],valid_stream:1,valu:[0,2,4,7,9,10,15],valueerror:[7,10,11],variabl:[0,2,5,7,9,10,15],variable_nam:10,variou:[0,1,8,15],vector:15,verb:10,verbos:6,veri:[1,4,15],verif:15,via:[9,10,14],visual:[0,1],wai:[0,6,13],want:15,warn:[0,7,10],wast:1,watch:[2,15],well:[0,10,11,15],were:[10,11],what:[0,2,4,5,15],when:[0,1,4,5,6,7,9,10,11,15],whenev:9,where:[0,2,4,5,15],wherein:[0,10],whether:[1,4,5,9,10],which:[0,1,2,4,5,6,8,9,10,11,13,15],whichev:10,whole:[0,1,2,4,5,6,15],width:0,wish:[1,5],within:10,without:[1,5,6,9,10],won:1,word:15,work:1,workflow:5,world:15,would:[0,1,2,5,8,9],wrap:15,wrapper:[9,11],write:[1,2,10],write_csv:10,writecsv:[2,10],writetensorboard:2,written:[2,9,10],wrong:10,xxx:10,y_hat:15,yaml:[0,1,8,9,15],yet:[0,5,10],yield:[1,15],you:[0,1,2,5,14,15],your:[0,1,2,4,15],zero:15,zeroth:7},titles:["Configuration","Dataset","Hooks","Advanced","Main Loop","Model","CLI Reference","<code class=\"docutils literal notranslate\"><span class=\"pre\">cxflow</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">cxflow.constants</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">cxflow.datasets</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">cxflow.hooks</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">cxflow.models</span></code>","API Reference","Getting Started","cxflow","Tutorial"],titleterms:{"class":[7,9,10,11],The:1,Using:15,addit:1,advanc:3,after_batch:2,after_epoch:2,api:12,argument:6,basedataset:1,cli:6,conclus:0,configur:[0,2,15],constant:8,contribut:14,cxflow:[6,7,8,9,10,11,14,15],data:1,dataset:[0,1,4,6,9,15],develop:13,evalu:[0,4],event:2,except:10,get:13,hook:[0,2,4,10,15],instal:13,integr:4,introduct:15,lazi:1,licens:14,lifecycl:4,loop:[0,4,15],main:[0,4,15],method:1,model:[0,4,5,11,15],name:6,philosophi:1,posit:6,predict:6,prune:6,refer:[6,12],regular:2,requir:13,restor:5,resum:6,run:5,start:13,stream:1,submodul:7,support:14,task:15,train:[4,6],tutori:15,variabl:8}})