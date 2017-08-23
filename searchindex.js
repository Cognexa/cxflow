Search.setIndex({docnames:["advanced/config","advanced/dataset","advanced/hook","advanced/index","advanced/main_loop","advanced/model","cxflow/cxflow","cxflow/cxflow.cli","cxflow/cxflow.constants","cxflow/cxflow.datasets","cxflow/cxflow.hooks","cxflow/cxflow.models","cxflow/cxflow.utils","cxflow/index","getting_started","index","tutorial"],envversion:53,filenames:["advanced/config.rst","advanced/dataset.rst","advanced/hook.rst","advanced/index.rst","advanced/main_loop.rst","advanced/model.rst","cxflow/cxflow.rst","cxflow/cxflow.cli.rst","cxflow/cxflow.constants.rst","cxflow/cxflow.datasets.rst","cxflow/cxflow.hooks.rst","cxflow/cxflow.models.rst","cxflow/cxflow.utils.rst","cxflow/index.rst","getting_started.rst","index.rst","tutorial.rst"],objects:{"":{cxflow:[6,0,0,"-"]},"cxflow.AbstractDataset":{__init__:[6,2,1,""],__weakref__:[6,3,1,""]},"cxflow.AbstractModel":{__init__:[6,2,1,""],__weakref__:[6,3,1,""],input_names:[6,3,1,""],output_names:[6,3,1,""],restore_fallback:[6,3,1,""],run:[6,2,1,""],save:[6,2,1,""]},"cxflow.BaseDataset":{__init__:[6,2,1,""],__weakref__:[6,3,1,""],_init_with_kwargs:[6,2,1,""]},"cxflow.MainLoop":{__init__:[6,2,1,""],__weakref__:[6,3,1,""],_check_sources:[6,2,1,""],_create_epoch_data:[6,2,1,""],_run_epoch:[6,2,1,""],_run_zeroth_epoch:[6,2,1,""],evaluate_stream:[6,2,1,""],get_stream:[6,2,1,""],run_prediction:[6,2,1,""],run_training:[6,2,1,""],train_by_stream:[6,2,1,""]},"cxflow.cli":{_build_grid_search_commands:[7,4,1,""],create_dataset:[7,4,1,""],create_hooks:[7,4,1,""],create_model:[7,4,1,""],create_output_dir:[7,4,1,""],fallback:[7,4,1,""],find_config:[7,4,1,""],get_cxflow_arg_parser:[7,4,1,""],grid_search:[7,4,1,""],invoke_dataset_method:[7,4,1,""],predict:[7,4,1,""],resume:[7,4,1,""],run:[7,4,1,""],train:[7,4,1,""],validate_config:[7,4,1,""]},"cxflow.constants":{CXF_CONFIG_FILE:[8,5,1,""],CXF_FULL_DATE_FORMAT:[8,5,1,""],CXF_HOOKS_MODULE:[8,5,1,""],CXF_LOG_DATE_FORMAT:[8,5,1,""],CXF_LOG_FILE:[8,5,1,""],CXF_LOG_FORMAT:[8,5,1,""]},"cxflow.datasets":{AbstractDataset:[9,1,1,""],BaseDataset:[9,1,1,""]},"cxflow.datasets.AbstractDataset":{__init__:[9,2,1,""],__weakref__:[9,3,1,""]},"cxflow.datasets.BaseDataset":{__init__:[9,2,1,""],__weakref__:[9,3,1,""],_init_with_kwargs:[9,2,1,""]},"cxflow.hooks":{AbstractHook:[10,1,1,""],AccumulateVariables:[10,1,1,""],CatchSigint:[10,1,1,""],Check:[10,1,1,""],ComputeStats:[10,1,1,""],LogProfile:[10,1,1,""],LogVariables:[10,1,1,""],SaveBest:[10,1,1,""],SaveEvery:[10,1,1,""],StopAfter:[10,1,1,""],TrainingTerminated:[10,7,1,""],WriteCSV:[10,1,1,""]},"cxflow.hooks.AbstractHook":{CXF_HOOK_INIT_ARGS:[10,3,1,""],__init__:[10,2,1,""],__weakref__:[10,3,1,""],after_batch:[10,2,1,""],after_epoch:[10,2,1,""],after_epoch_profile:[10,2,1,""],after_training:[10,2,1,""],before_training:[10,2,1,""]},"cxflow.hooks.AccumulateVariables":{__init__:[10,2,1,""],_reset_accumulator:[10,2,1,""],after_batch:[10,2,1,""],after_epoch:[10,2,1,""]},"cxflow.hooks.CatchSigint":{_sigint_handler:[10,2,1,""],after_batch:[10,2,1,""],after_training:[10,2,1,""],before_training:[10,2,1,""]},"cxflow.hooks.Check":{__init__:[10,2,1,""],after_epoch:[10,2,1,""]},"cxflow.hooks.ComputeStats":{AGGREGATIONS:[10,3,1,""],__init__:[10,2,1,""],_compute_aggregation:[10,6,1,""],_save_stats:[10,2,1,""],after_epoch:[10,2,1,""]},"cxflow.hooks.LogProfile":{after_epoch_profile:[10,2,1,""]},"cxflow.hooks.LogVariables":{__init__:[10,2,1,""],_log_variables:[10,2,1,""],after_epoch:[10,2,1,""]},"cxflow.hooks.SaveBest":{CONDITIONS:[10,3,1,""],__init__:[10,2,1,""],_get_value:[10,2,1,""],_is_value_better:[10,2,1,""],after_epoch:[10,2,1,""]},"cxflow.hooks.SaveEvery":{SAVE_FAILURE_ACTIONS:[10,3,1,""],__init__:[10,2,1,""],after_epoch:[10,2,1,""],save_model:[10,6,1,""]},"cxflow.hooks.StopAfter":{__init__:[10,2,1,""],after_epoch:[10,2,1,""]},"cxflow.hooks.WriteCSV":{MISSING_VARIABLE_ACTIONS:[10,3,1,""],UNKNOWN_TYPE_ACTIONS:[10,3,1,""],__init__:[10,2,1,""],_write_header:[10,2,1,""],_write_row:[10,2,1,""],after_epoch:[10,2,1,""]},"cxflow.models":{AbstractModel:[11,1,1,""]},"cxflow.models.AbstractModel":{__init__:[11,2,1,""],__weakref__:[11,3,1,""],input_names:[11,3,1,""],output_names:[11,3,1,""],restore_fallback:[11,3,1,""],run:[11,2,1,""],save:[11,2,1,""]},"cxflow.utils":{DisabledLogger:[12,1,1,""],DisabledPrint:[12,1,1,""],Timer:[12,1,1,""],_EMPTY_DICT:[12,5,1,""],config_to_file:[12,4,1,""],config_to_str:[12,4,1,""],create_object:[12,4,1,""],find_class_module:[12,4,1,""],get_class_module:[12,4,1,""],list_submodules:[12,4,1,""],load_config:[12,4,1,""],parse_arg:[12,4,1,""],parse_fully_qualified_name:[12,4,1,""]},"cxflow.utils.DisabledLogger":{__enter__:[12,2,1,""],__exit__:[12,2,1,""],__weakref__:[12,3,1,""]},"cxflow.utils.DisabledPrint":{__enter__:[12,2,1,""],__exit__:[12,2,1,""],__weakref__:[12,3,1,""]},"cxflow.utils.Timer":{__enter__:[12,2,1,""],__exit__:[12,2,1,""],__init__:[12,2,1,""],__weakref__:[12,3,1,""]},cxflow:{AbstractDataset:[6,1,1,""],AbstractModel:[6,1,1,""],BaseDataset:[6,1,1,""],MainLoop:[6,1,1,""],cli:[7,0,0,"-"],constants:[8,0,0,"-"],datasets:[9,0,0,"-"],hooks:[10,0,0,"-"],models:[11,0,0,"-"],utils:[12,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"],"5":["py","data","Python data"],"6":["py","staticmethod","Python static method"],"7":["py","exception","Python exception"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function","5":"py:data","6":"py:staticmethod","7":"py:exception"},terms:{"06d":8,"0x7efc1877eae8":[],"0x7efca7139598":[],"0x7efd3491cae8":[],"0x7efd425699d8":[],"0x7efd56d02510":[],"0x7efe4ef5a598":[],"0x7efe6adb99d8":[],"0x7eff7f2ed598":[],"0x7effa7f319d8":[],"0x7effca8059d8":[],"0x7f00539e6598":[],"0x7f00afaf4ae8":[],"0x7f0100646ae8":[],"0x7f0165337ae8":[],"0x7f02ee4f9ae8":[],"0x7f03057d0ae8":[],"0x7f038fe4b598":[],"0x7f03b453c9d8":[],"0x7f03c349d9d8":[],"0x7f042b94dae8":[],"0x7f0443fdaae8":[],"0x7f04a4fa89d8":[],"0x7f051499eae8":[],"0x7f054741b598":[],"0x7f0567b829d8":[],"0x7f058d53d9d8":[],"0x7f06146ab598":[],"0x7f06dcc5eae8":[],"0x7f0742a64598":[],"0x7f08a2f96ae8":[],"0x7f08a33839d8":[],"0x7f08a880fae8":[],"0x7f0a3913c9d8":[],"0x7f0a45991598":[],"0x7f0ab4e68598":[],"0x7f0b3dfc2ae8":[],"0x7f0b51491ae8":[],"0x7f0bde2fdae8":[],"0x7f0c27fa0ae8":[],"0x7f0c52efe9d8":[],"0x7f0d8a966598":[],"0x7f0d8b886598":[],"0x7f0e8b7c7ae8":[],"0x7f0eb6ccdae8":[],"0x7f0f555619d8":[],"0x7f0ff8f16ae8":[],"0x7f0ffe6f99d8":[],"0x7f121c041598":[],"0x7f14202779d8":[],"0x7f1465a72598":[],"0x7f146863fae8":[],"0x7f149aac9ae8":[],"0x7f1556e82ae8":[],"0x7f1747edfae8":[],"0x7f1780701ae8":[],"0x7f17e20cf598":[],"0x7f18659ffae8":[],"0x7f18711c8598":[],"0x7f194cac0598":[],"0x7f1a11cf2598":[],"0x7f1a29db7598":[],"0x7f1a37e40598":[],"0x7f1a60e54ae8":[],"0x7f1aff5ecae8":[],"0x7f1b7efaf9d8":[],"0x7f1cac3d2ae8":[],"0x7f1d4870c510":[],"0x7f1d4ab449d8":[],"0x7f1e0091aae8":[],"0x7f1f6e3f49d8":[],"0x7f2023698ae8":[],"0x7f20ccaaf9d8":[],"0x7f20d1ee4ae8":[],"0x7f215a749598":[],"0x7f21772f6ae8":[],"0x7f21c0138598":[],"0x7f21c8ebcae8":[],"0x7f228ab70ae8":[],"0x7f233f56d598":[],"0x7f236267a598":[],"0x7f2471b14598":[],"0x7f24fb9ceae8":[],"0x7f26506619d8":[],"0x7f269218f598":[],"0x7f2717f9cae8":[],"0x7f27979459d8":[],"0x7f27b1295598":[],"0x7f27b74f6ae8":[],"0x7f27cba2c598":[],"0x7f285b9fc598":[],"0x7f2863f8e598":[],"0x7f2873f1cae8":[],"0x7f28855a09d8":[],"0x7f28872fa9d8":[],"0x7f29be6a59d8":[],"0x7f29eccaf9d8":[],"0x7f2a54e93598":[],"0x7f2a59cb69d8":[],"0x7f2ab46209d8":[],"0x7f2b98a079d8":[],"0x7f2c0d9d29d8":[],"0x7f2c3e30d9d8":[],"0x7f2c8140dae8":[],"0x7f2d4fa42ae8":[],"0x7f2d89280ae8":[],"0x7f2dc4017ae8":[],"0x7f2e9c5209d8":[],"0x7f2f085b4ae8":[],"0x7f3036c3a598":[],"0x7f308b177598":[],"0x7f31201ee9d8":[],"0x7f31216ebae8":[],"0x7f3138a2dae8":[],"0x7f31f0c0a598":[],"0x7f32ba063ae8":[],"0x7f32f356d598":[],"0x7f3362e6ca60":[],"0x7f340501fae8":[],"0x7f340ae92ae8":[],"0x7f34ded87ae8":[],"0x7f3676663ae8":[],"0x7f36c09ec9d8":[],"0x7f3769f22ae8":[],"0x7f3774f33ae8":[],"0x7f3822f6bae8":[],"0x7f39a1bb7598":[],"0x7f3a09b03ae8":[],"0x7f3a199069d8":[],"0x7f3a84a63ae8":[],"0x7f3a8ff72ae8":[],"0x7f3aaa889ae8":[],"0x7f3b3e0dbae8":[],"0x7f3c1991b598":[],"0x7f3ca0b279d8":[],"0x7f3cd398c598":[],"0x7f3ce70a2ae8":[],"0x7f3d388489d8":[],"0x7f3dd21d0ae8":[],"0x7f3e886feae8":[],"0x7f4046edb598":[],"0x7f40586e39d8":[],"0x7f410c352ae8":[],"0x7f4177b65ae8":[],"0x7f42294d49d8":[],"0x7f42de1c6ae8":[],"0x7f432462a9d8":[],"0x7f4376deaae8":[],"0x7f44afce8ae8":[],"0x7f44bde6b598":[],"0x7f44c9f60ae8":[],"0x7f455fedaae8":[],"0x7f456ae18ae8":[],"0x7f4607772598":[],"0x7f4702c63ae8":[],"0x7f47112339d8":[],"0x7f47c6ed6ae8":[],"0x7f47d4007598":[],"0x7f47e5503ae8":[],"0x7f481165c598":[],"0x7f489d249ae8":[],"0x7f49ce3d0598":[],"0x7f49eaf18ae8":[],"0x7f49fa7af598":[],"0x7f4a00689598":[],"0x7f4a4443c598":[],"0x7f4a9d985598":[],"0x7f4b0c870ae8":[],"0x7f4b17f21ae8":[],"0x7f4b5d403ae8":[],"0x7f4baaeeaae8":[],"0x7f4c8b3bfa60":[],"0x7f4cf2bca9d8":[],"0x7f4d2c008598":[],"0x7f4e486209d8":[],"0x7f4ea80df598":[],"0x7f4ef158c9d8":[],"0x7f4fa7b5dae8":[],"0x7f506543b9d8":[],"0x7f50dea49598":[],"0x7f51804c8ae8":[],"0x7f518d3829d8":[],"0x7f51987d19d8":[],"0x7f530bcaa9d8":[],"0x7f532544cae8":[],"0x7f53273529d8":[],"0x7f538efe99d8":[],"0x7f53efda8598":[],"0x7f5446f06ae8":[],"0x7f5479340ae8":[],"0x7f54be4329d8":[],"0x7f55088679d8":[],"0x7f5509988598":[],"0x7f556617e9d8":[],"0x7f5578f4d598":[],"0x7f557d7c39d8":[],"0x7f5623261ae8":[],"0x7f5653bebae8":[],"0x7f565a807ae8":[],"0x7f565ec78a60":[],"0x7f5694685ae8":[],"0x7f56b8e1c598":[],"0x7f56d0836ae8":[],"0x7f5aac414ae8":[],"0x7f5bbd1e8ae8":[],"0x7f5bfaf2cae8":[],"0x7f5c2cb68ae8":[],"0x7f5c3af46ae8":[],"0x7f5c9fe049d8":[],"0x7f5cda002ae8":[],"0x7f5cdab64ae8":[],"0x7f5d5a0129d8":[],"0x7f5d9981cae8":[],"0x7f5eced52598":[],"0x7f5ef1658ae8":[],"0x7f5fea56a598":[],"0x7f60467ac9d8":[],"0x7f6072c05598":[],"0x7f60fb460ae8":[],"0x7f610dddaae8":[],"0x7f62bab9c598":[],"0x7f63aa637598":[],"0x7f63f6890598":[],"0x7f64f7176598":[],"0x7f66a27ab598":[],"0x7f679ecc1ae8":[],"0x7f68f14b3ae8":[],"0x7f694d1a0598":[],"0x7f69fc955510":[],"0x7f6ab24ad9d8":[],"0x7f6af5965ae8":[],"0x7f6b56462598":[],"0x7f6c0c97bae8":[],"0x7f6d10106ae8":[],"0x7f6da53a0ae8":[],"0x7f6da76e2ae8":[],"0x7f6e55dd4ae8":[],"0x7f6e89999ae8":[],"0x7f6f02af8ae8":[],"0x7f6f03a3dae8":[],"0x7f6fe76caae8":[],"0x7f70493aa598":[],"0x7f70afe1aae8":[],"0x7f70cf9c5598":[],"0x7f70cff0cae8":[],"0x7f720fbad9d8":[],"0x7f72e4ba3ae8":[],"0x7f72e80dc9d8":[],"0x7f72ec429598":[],"0x7f72f955f598":[],"0x7f7316b7aae8":[],"0x7f753947d9d8":[],"0x7f755da38598":[],"0x7f7584430598":[],"0x7f76386f0598":[],"0x7f76937c4ae8":[],"0x7f772d043598":[],"0x7f7872badae8":[],"0x7f78781e3598":[],"0x7f789f6bd598":[],"0x7f78b5f49ae8":[],"0x7f79c35c1f28":[],"0x7f7a58329598":[],"0x7f7a95398598":[],"0x7f7b7f41aae8":[],"0x7f7bd0514ae8":[],"0x7f7c28f179d8":[],"0x7f7c9e30bae8":[],"0x7f7cc181fae8":[],"0x7f7d3f50bae8":[],"0x7f7db0b78598":[],"0x7f7e0f9d19d8":[],"0x7f7e1ba01598":[],"0x7f7e6f68c598":[],"0x7f7f9c85eae8":[],"0x7f8024163ae8":[],"0x7f80fad2dae8":[],"0x7f815ade39d8":[],"0x7f816dcd9ae8":[],"0x7f824e6bb598":[],"0x7f82b623a598":[],"0x7f83445e8ae8":[],"0x7f83f0c089d8":[],"0x7f844ae38ae8":[],"0x7f859d9de598":[],"0x7f85e2609ae8":[],"0x7f8637acbae8":[],"0x7f863b6a6ae8":[],"0x7f864b8fa9d8":[],"0x7f8698a06598":[],"0x7f8960256ae8":[],"0x7f89bfd55ae8":[],"0x7f8adca80ae8":[],"0x7f8adcc719d8":[],"0x7f8b12e69ae8":[],"0x7f8b28d059d8":[],"0x7f8c4f2adae8":[],"0x7f8c838b7ae8":[],"0x7f8d008679d8":[],"0x7f8d53a65ae8":[],"0x7f8e39614ae8":[],"0x7f8f87909598":[],"0x7f914c9ca598":[],"0x7f91bc2e3598":[],"0x7f92c52f1ae8":[],"0x7f93487f5598":[],"0x7f9349737ae8":[],"0x7f93857809d8":[],"0x7f95b13459d8":[],"0x7f9683d43ae8":[],"0x7f984d883ae8":[],"0x7f9888b589d8":[],"0x7f9a17159ae8":[],"0x7f9a1c647598":[],"0x7f9a44df8ae8":[],"0x7f9a814eb598":[],"0x7f9a8bc38ae8":[],"0x7f9ab5f19ae8":[],"0x7f9b107779d8":[],"0x7f9b5dc269d8":[],"0x7f9d9d3299d8":[],"0x7f9dc8217598":[],"0x7f9df7c60ae8":[],"0x7f9e0bd07598":[],"0x7f9e34fd5ae8":[],"0x7f9e5eb86598":[],"0x7f9ed5935ae8":[],"0x7f9f40be2ae8":[],"0x7f9f6bb42598":[],"0x7f9fd89ee598":[],"0x7fa1792dbae8":[],"0x7fa37d759ae8":[],"0x7fa49ead4ae8":[],"0x7fa52e00b598":[],"0x7fa59f0e9ae8":[],"0x7fa5b7e4e598":[],"0x7fa5ed94b598":[],"0x7fa694029ae8":[],"0x7fa76f052ae8":[],"0x7fa8bdaf4598":[],"0x7fa987ef19d8":[],"0x7faa046e0510":[],"0x7fabd1262ae8":[],"0x7fac7e6d3ae8":[],"0x7fac8efebae8":[],"0x7fada5e58ae8":[],"0x7fae3b505598":[],"0x7faeae8989d8":[],"0x7faff5248a60":[],"0x7fb1518a7598":[],"0x7fb1a1d8f598":[],"0x7fb2121c6598":[],"0x7fb28c51c598":[],"0x7fb2b6fc2ae8":[],"0x7fb379b4f598":[],"0x7fb42ca15598":[],"0x7fb42de8dae8":[],"0x7fb436c51ae8":[],"0x7fb44ff969d8":[],"0x7fb471e289d8":[],"0x7fb4a7a65ae8":[],"0x7fb62c4cfae8":[],"0x7fb6540baae8":[],"0x7fb6597499d8":[],"0x7fb76d5ed9d8":[],"0x7fb77ea94ae8":[],"0x7fb78de73ae8":[],"0x7fb95248cae8":[],"0x7fbb86e899d8":[],"0x7fbbc77b5ae8":[],"0x7fbc32bdc598":[],"0x7fbc84fd4598":[],"0x7fbc9824e9d8":[],"0x7fbcb6fdaae8":[],"0x7fbd03616ae8":[],"0x7fbd0f496598":[],"0x7fbd286fb9d8":[],"0x7fbf5b349ae8":[],"0x7fc023cf9598":[],"0x7fc11aaa39d8":[],"0x7fc167e7cae8":[],"0x7fc18dd50ae8":[],"0x7fc18f649ae8":[],"0x7fc36152c9d8":[],"0x7fc3d7592598":[],"0x7fc418047ae8":[],"0x7fc454c14598":[],"0x7fc487fc09d8":[],"0x7fc5bb21f9d8":[],"0x7fc5bbd219d8":[],"0x7fc75fa9cae8":[],"0x7fc78381f598":[],"0x7fc92c8229d8":[],"0x7fc98e615598":[],"0x7fc9bfebcae8":[],"0x7fc9c405d9d8":[],"0x7fc9faa489d8":[],"0x7fca7cdb0ae8":[],"0x7fca93baaae8":[],"0x7fcacd79c598":[],"0x7fcb86060ae8":[],"0x7fcbcb5c49d8":[],"0x7fcbd1bf7598":[],"0x7fcc011b19d8":[],"0x7fcc3f057ae8":[],"0x7fcc56935598":[],"0x7fcc6933c598":[],"0x7fcce12e9598":[],"0x7fcd39d5a9d8":[],"0x7fcd6505cae8":[],"0x7fce8a05b598":[],"0x7fcec693f9d8":[],"0x7fcf71f159d8":[],"0x7fcfb4f79ae8":[],"0x7fd124df6598":[],"0x7fd136db1598":[],"0x7fd1544d99d8":[],"0x7fd176b8aae8":[],"0x7fd1d9242598":[],"0x7fd1e9053598":[],"0x7fd2cdb57ae8":[],"0x7fd3ac7b0ae8":[],"0x7fd4ec7759d8":[],"0x7fd546f8aae8":[],"0x7fd5622db598":[],"0x7fd5b0e0eae8":[],"0x7fd5b8b0bae8":[],"0x7fd5c40faae8":[],"0x7fd5d707cae8":[],"0x7fd6d9230598":[],"0x7fd77d669ae8":[],"0x7fd7dc550ae8":[],"0x7fd833937598":[],"0x7fd858003ae8":[],"0x7fd85aea1598":[],"0x7fd96d0a1ae8":[],"0x7fd9adb50598":[],"0x7fd9c1ccdae8":[],"0x7fd9da7baae8":[],"0x7fd9e9f86ae8":[],"0x7fd9ec01f9d8":[],"0x7fdaa3f11ae8":[],"0x7fdab99e09d8":[],"0x7fdb37c3eae8":[],"0x7fdb4ba1aae8":[],"0x7fdb8fb54598":[],"0x7fdbb955a598":[],"0x7fdc13bfc9d8":[],"0x7fdc53af6a60":[],"0x7fdcceb1e598":[],"0x7fdd1b6b6ae8":[],"0x7fdd5d608598":[],"0x7fde459db598":[],"0x7fdf4691b598":[],"0x7fe04df18ae8":[],"0x7fe066c99598":[],"0x7fe09146fae8":[],"0x7fe10ece9ae8":[],"0x7fe1d2d929d8":[],"0x7fe205998598":[],"0x7fe213523598":[],"0x7fe24a256ae8":[],"0x7fe2865219d8":[],"0x7fe28d2b6598":[],"0x7fe2dc482598":[],"0x7fe323b0bae8":[],"0x7fe34d858ae8":[],"0x7fe3ae3ba598":[],"0x7fe40f0e1598":[],"0x7fe442b04ae8":[],"0x7fe49f42dae8":[],"0x7fe4dbfb1ae8":[],"0x7fe5b7186ae8":[],"0x7fe5d88e6598":[],"0x7fe72cc199d8":[],"0x7fe893bcf598":[],"0x7fe8d0e749d8":[],"0x7fe9e03df598":[],"0x7fedaaf2a9d8":[],"0x7fedb3da2598":[],"0x7fedb48caae8":[],"0x7fee4d404ae8":[],"0x7feecce1cae8":[],"0x7feef2bdc9d8":[],"0x7fef14c6b598":[],"0x7fefd434eae8":[],"0x7ff06cdf7ae8":[],"0x7ff11ed849d8":[],"0x7ff144eaaae8":[],"0x7ff1db7ea598":[],"0x7ff2cddaa598":[],"0x7ff34f3ccae8":[],"0x7ff3563d3ae8":[],"0x7ff4b3c8bae8":[],"0x7ff4eb5beae8":[],"0x7ff6c8a2fae8":[],"0x7ff75eaa3598":[],"0x7ff838f11598":[],"0x7ff83b8169d8":[],"0x7ff98bd51598":[],"0x7ffa6b1c2ae8":[],"0x7ffac24c0ae8":[],"0x7ffb19ff7ae8":[],"0x7ffb3bedba60":[],"0x7ffb5040f598":[],"0x7ffb8f6c8ae8":[],"10th":10,"15s":8,"1st":2,"2nd":2,"800x600":1,"800x600x3":1,"abstract":[6,11],"boolean":[4,5],"case":[0,1,2,5,7,10,16],"catch":10,"class":[0,1,2,5,7,16],"default":[0,4,7,10],"final":[0,1,4,14,16],"float":[10,12],"function":[1,4,6,10,16],"import":[0,1,2,5,8,12,16],"int":[1,2,6,7,10,16],"new":[0,2,5,6,9,10,12,16],"null":12,"return":[4,5,6,7,10,11,12,16],"short":2,"static":10,"super":2,"switch":10,"true":[0,1,5,6,7,11],"try":16,"var":0,"while":[0,1,4,5],For:[0,1,2,5,10,16],One:[4,10],THE:1,That:[1,16],The:[0,2,3,4,5,6,7,9,10,14,15,16],There:4,These:[0,5],With:15,__enter__:12,__exit__:12,__init__:[2,6,9,10,11,12],__str__:[],__weakref__:[6,9,10,11,12],_accumul:10,_build_grid_search_command:7,_check_sourc:6,_compute_aggreg:10,_create_epoch_data:6,_create_model:16,_create_net:16,_empty_dict:12,_end:[],_epoch_limit:[2,10],_get_valu:10,_init_with_kwarg:[1,6,9,16],_is_value_bett:10,_log_vari:10,_name:12,_profil:12,_reset_accumul:10,_run_epoch:6,_run_zeroth_epoch:6,_save_every_n_epoch:10,_save_stat:10,_session:16,_sigint_handl:10,_start:12,_stream:[1,6,9,10,16],_variabl:10,_write_head:10,_write_row:10,abl:[1,5,12,16],about:[5,7,10,12],abov:[0,1,2,5,16],abstract_dataset:[6,9],abstract_hook:[2,10],abstract_model:[6,11],abstractdataset:[6,7,9,11,16],abstracthook:[2,6,7,10],abstractmodel:[5,6,7,10,11],accept:[1,2,5,16],access:16,accord:[6,7,11],accumul:[2,10],accumulate_vari:10,accumulatevari:[2,10],accuraci:[0,2,10,16],act:2,action:[6,10,12,16],actual:[1,2],adam:16,adamoptim:16,add:[6,9,16],addit:[0,2,3,4,6,7,9,10,11,12,14,16],addition:[5,6,7,9,10],additiona:10,additional_arg:12,advanc:[0,15,16],advantag:0,after:[0,2,4,7,10,16],after_batch:[3,10],after_batch_hooks_train:10,after_epoch:[3,10],after_epoch_hook:10,after_epoch_profil:[2,10],after_train:[2,10],afterward:4,again:[0,16],aggreag:10,aggreg:10,aka:16,algorithm:16,all:[0,1,2,5,6,7,10,11,12,16],allow:[2,7,10,16],almost:0,along:[1,4],alreadi:[0,1,2,7,16],also:[2,4,5,6,11,16],altern:4,although:5,altogeth:2,analog:[0,1],anchor:0,ani:[1,2,5,6,7,9,10,12,16],anim:[0,1,5],annoi:1,anoth:0,answer:16,anyth:16,anywai:10,api:[6,9,10,15],append:[6,11,12],approach:0,approch:0,appropri:0,arbitrari:[1,16],arbitrarili:0,aren:0,arg1:2,arg2:2,arg:[2,7,10,12],argmax:16,argument:[0,1,2,4,5,7,10,12,16],argumentpars:7,around:16,artifact:16,artifici:1,asctim:8,assert:7,assum:1,attempt:5,attribut:[0,16],attributeerror:6,augment:[0,1],automat:[0,1,5,14,16],avail:[2,6,7,9,10],averag:10,avoid:16,axi:16,back:[0,12],backend:[4,5,6,7,11,14],base:[4,6,9,10,11,12,16],base_dataset:[6,9],baseclass:[5,6,11],basedataset:[3,6,9,16],basemodel:[5,16],basic:16,batch:[0,1,2,4,5,6,10,11,16],batch_data:[2,10],batch_siz:[0,1,16],been:[2,4],befor:[0,2,4,10,14,16],before_train:[2,10],begin:16,behavior:[0,2,5,12],being:[2,10],below:[14,16],best:[1,2,10,16],bestsaverhook:[10,16],better:[10,16],between:[2,12],big:[1,16],bit:16,blank:10,block:16,blur:[0,1],blur_prob:[0,1],bool:[6,7,10,11,16],both:[0,5,7],branch:14,broken:13,buffer:[],build:[0,4,7,14,16],built:[2,7],bytes_or_buff:[],caffe2:16,calcul:16,call:[2,4,5,10,16],can:[0,1,2,4,15,16],cannot:5,carefulli:2,cast:16,cat:1,catch_sigint:10,catcher:10,catchsigint:[2,10],caught:10,caus:7,certain:10,chang:[0,1,16],check:[1,6,7,10,15],checkpoint:[5,7],checksum:1,child:10,cio:7,cl_argument:7,class_nam:12,classif:[1,2],classifi:[0,1],classificationinfohook:[0,2],clear:0,cli:[1,6,12,13],clon:14,clone:14,closer:2,cntk:16,code:[5,10,12,14,15],cognexa:[6,14,15],collect:[1,7,10],column:10,com:[14,15],combin:0,comfort:0,command:[1,4,5,7,12,16],comment:[0,1],common:0,complet:[0,1,16],compon:[0,3,5,15,16],comput:[0,2,5,10,16],compute_stat:10,computestat:[0,2,10],concept:[1,6,9],conclus:3,condit:10,config:[0,1,2,4,6,7,8,9,10,12,16],config_fil:[7,12],config_path:7,config_str:[6,9],config_to_fil:12,config_to_str:12,configur:[1,3,5,6,7,8,9,11,12],consequ:16,consid:5,consist:[1,7,16],constant:[6,13,16],construct:[0,1,5,6,10,11,16],constructor:[0,1,2,4,5,6,7,9,11,12,16],consum:1,contain:[0,1,2,5,6,7,8,9,10,12,16],continu:5,contrari:16,contrib:16,conveni:[0,1],convent:[10,16],convert:16,core:4,correct:16,correspond:[1,4,5,16],could:[1,12],counter:10,cover:[6,11,16],creat:[1,2,4,5,6,7,9,10,12,16],create_dataset:7,create_hook:7,create_model:7,create_object:12,create_optim:16,create_output_dir:7,creation:[2,7,16],csv:[2,10,16],ctrl:2,current:[0,6,10,11,16],custom:[7,16],custum:0,cxf_config_fil:[7,8],cxf_full_date_format:8,cxf_hook_init_arg:10,cxf_hooks_modul:8,cxf_log_date_format:8,cxf_log_fil:8,cxf_log_format:8,cxflow:[0,1,2,3,4,5,13,14],cxflow_scikit:[0,2],cxflow_tensorflow:[0,2,5,7,16],data:[0,2,3,4,6,10,16],data_root:0,databas:[0,16],databs:1,dataset:[2,3,5,6,7,10,11,13],dataset_log:7,datat:10,date:8,datset:0,debug:16,decid:2,decod:[1,6,9,16],deep:[0,15],def:[1,2,16],default_model_nam:7,default_valu:10,defaultdict:10,defin:[0,1,5,6,9,10,11,12,15,16],delet:5,delimit:10,demonstr:[1,5,16],denot:16,dens:16,depend:[4,14],depict:1,deploi:0,deriv:[2,7],describ:[1,4,10,16],descript:16,design:16,desir:16,detail:[2,3,4,16],detect:10,determin:[5,10],dev:[12,14],develop:[0,1,15],deviat:16,dict:[0,1,2,6,7,10,11,12],dictionari:[5,10,16],differ:[0,4,5,12],dig:[2,15,16],dim:16,dimens:16,dir:[6,7,9,11],directli:[0,10,14,16],directori:[0,2,5,6,7,8,9,10,11,12,14,16],disabl:12,disabledlogg:12,disabledprint:12,distinct:0,dive:[3,14],divis:10,document:[0,16],doe:[0,4,6,10,16],doesn:[0,1],dog:1,don:[0,1,16],done:[0,5,16],dot:[0,12],doubl:[6,10],down:13,download:1,drive:1,dry_run:7,dtype:16,dump:[5,7,8],duplic:0,dure:[0,1,2,4,10,16],each:[0,2,4,5,10,16],easi:16,easier:16,easili:[0,1,15,16],either:[0,1,7,10,15],eleg:1,els:16,embed:0,emploi:[0,1,16],empti:[2,6,10],enabl:[0,1],encapsul:1,encod:[0,1,7,16],encourag:16,end:[4,12],enforc:0,enough:16,enter:[2,10],entir:12,entri:[0,2,5,6,7,9,10,12],entry_point:7,enumer:10,environ:[5,16],environemt:5,epoch:[0,2,4,6,10,16],epoch_data:[2,10],epoch_id:[2,5,10],epoch_limit:[2,16],epoch_profil:2,epochstopperhook:[2,16],equal:16,error:[0,6,10,12,16],estim:16,etc:[1,4,5,6],eval_batch_train:10,evalu:[0,1,4,5,6,16],evaluate_stream:6,evaluet:0,even:[2,4],event:[3,4,7,10,12,16],everi:[2,5,6,9,10,16],everyth:[2,16],eviron:16,exactli:[0,10],exampl:[0,1,2,5,10,15,16],exce:10,exceed:10,except:[7,12],execut:[1,4,7,10,12,16],exist:[5,6,7,11],exit:10,expect:[0,1,2,5,7,10,16],experi:[0,1],explain:16,explicit:16,expos:[6,11],extend:[10,12,16],extra:[1,4,16],extra_stream:[0,1,4,6,10,16],extrem:[1,16],fact:2,fail:[6,7,10,11],failur:[7,10],fallback:[6,7,11],fals:[5,6,7,11],far:10,favourit:16,featur:[0,16],fed:[0,1,4],feed:[6,11],feel:16,fetch:[1,16],few:[0,16],figur:1,file:[0,1,2,5,6,7,8,10,11,12,14,16],filenam:12,filesystem:5,finali:16,find:12,find_class_modul:12,find_config:7,fine:[0,4],finish:[2,7,10,16],first:[0,1,2,4,5,10,16],fixed_batch_s:[0,6],float32:16,flow:10,focus:16,follow:[0,1,2,3,4,5,7,12,13,14,16],form:[0,1,7,10,16],format:[2,8,10,12],former:1,forward:[6,11],found:[4,7,10,12,16],four:[1,16],fq_name:12,fraction:16,framework:[2,16],free:16,from:[0,1,2,4,5,6,7,9,10,12,14,16],fscore:10,full:[8,12],fulli:[0,6,11,12],fundament:16,furhtermor:0,further:[0,1],futur:[15,16],gener:[0,1,8,16],get:[2,6,12,16],get_class_modul:12,get_cxflow_arg_pars:7,get_stream:6,getdefaultencod:[],git:14,github:[14,15],given:[0,1,2,5,6,7,9,10,11,12,16],global_variables_initi:16,goal:10,going:16,gold_vari:[0,2],good:[2,16],graph:16,greater_equ:16,grid:7,grid_search:7,ground:0,handl:[1,2,10],handler:10,happen:[4,16],has:[1,2,4,5,6,9,12],have:[0,2,6,9,10,12,15,16],header:10,height:[0,1],hello:7,helper:12,henc:[0,1,2,16],here:[5,12],hidden:[0,1,16],hidden_activ:16,home:6,hook:[3,5,6,7,8,13],how:[2,10,16],howev:[0,1,2,5],http:[14,15],idea:16,ident:0,ids:0,ignor:[0,1,6,10,12],imag:[0,1,2,5],imagin:2,img1:1,img2:1,img3:1,img4:1,immedi:[0,10],implement:[1,2,3,4,5,6,9,11,16],imposs:0,includ:[0,1,7,10,16],incorrectli:0,increas:10,index:10,indic:4,individu:[0,16],infer:[1,3,5,7,10],inference_logging_hook:0,inferencelogginghook:0,info:[2,7],inform:[0,3,4,5,6,7,11,16],inherit:[1,6,9,10],init:10,initi:[2,6,9],initil:16,input:[0,1,5,6,10,11,16],input_nam:[5,6,11],insid:[2,14],instal:16,instanc:[0,1,2,5,7,12],instanti:5,instead:[0,1,6,7,9,12,16],instruct:2,instrument:2,integr:[3,16],interest:1,interfac:[0,5,6,10,11,16],internet:1,interrupt:[2,4,10],introduc:[0,16],invoc:10,invok:[0,1,2,7,16],invoke_dataset_method:7,ioerror:10,issu:[4,15],item:[4,16],iter:[1,4,6,7,10,11,12,16],its:[0,1,2,4,5,7,10,16],itself:[0,4,10],join:7,json:[0,16],just:[0,5],keep:[1,2],kei:[0,1,5,12,16],kera:16,keyerror:10,know:[0,1,5,15,16],kwarg:[0,1,2,6,9,10,11,12,16],label:[0,2],languag:0,last:2,later:[10,16],latter:1,layer:16,lazi:3,learn:[0,2,5,6,11,15,16],learning_r:[0,16],least:10,leav:[1,7],len:16,length:16,less:0,let:[1,2,15,16],level:0,levelnam:8,lifecycl:[3,10],like:[2,10,16],line:[0,7,10],link:0,list:[0,2,5,6,7,9,10,11,12,14,15,16],list_submodul:12,littl:16,load:[0,1,5,6,7,12,16],load_config:12,load_training_batch:1,local:[],local_variables_initi:16,locat:[1,7,15],log:[0,2,5,6,7,8,9,10,12,16],log_dir:[6,7,11],log_profil:[0,10],log_vari:10,logger:[7,12],logginghook:16,logprofil:[0,2,10],logvari:[2,7,10],longer:0,look:[1,2,5,16],loop:[2,3,5,6,7,10],loss:[0,2,5,10,16],lot:16,luckili:2,machin:[0,5,6,11,16],mai:[0,1,2,5,6,7,9,10,12,16],mail:15,main:[1,2,3,6,7,10],main_loop:[0,1,2,4,6,7,16],mainloop:[5,6,7,9,11,16],major:[2,14,16],majority_dataset:16,majority_net:16,majoritydataset:16,majorityexampl:16,majorityexample_:16,majoritynet:16,make:[6,9,16],manag:[2,5,6,15],mandatori:0,manual:10,map:[1,6,10,11],mappingproxi:12,match:0,matter:[2,4],max:10,max_epoch:10,maximum:10,mean:[0,2,10,16],meant:10,measur:[0,10,12],median:10,memori:5,messag:[7,8],method:[0,2,3,4,5,6,7,9,10,11,12,16],method_nam:7,metric:[0,10],might:[0,1,4,5,14,16],mimic:12,min:10,minim:[10,16],minut:15,misc:12,miss:[6,10],missing_variable_act:10,mit:15,mix:0,mlp:16,mode:6,model:[1,2,3,6,7,10,13,15],modifi:[2,16],modul:[0,1,2,6,8,9,10,11,12,16],modular:[0,16],module_nam:12,moment:[1,10,16],monitor:10,more:[0,2,3,4,16],moreov:12,most:[5,10,14],motiv:0,msec:8,multipl:[4,10,12,16],must:[0,6,9,10,14,16],my_data:0,my_dataset:[0,1],my_hook:0,my_logger_nam:12,my_model:0,my_modul:0,my_project:2,my_work:12,mydataset:[0,1],myhook:[0,2],mymodel:0,n_epoch:[7,10],n_hidden_neuron:0,name:[0,1,2,4,5,6,7,8,10,11,12,14,16],name_suffix:[5,6,10,11],natur:[0,10,16],necessari:[1,16],need:[0,1,2,5,16],nest:[0,10],net:[4,16],network:[0,16],neural:0,neuron:0,never:4,nevertheless:[0,14],new_typ:[],new_valu:10,newli:16,newtyp:[],next:16,nightli:14,node:16,non:12,nonamemodel:7,none:[2,6,7,9,10,11,12,16],note:[0,1,2,5,10,16],noth:[2,10],now:[0,2,16],npr:16,num:1,number:[0,2,4,10,16],numerical_param:7,numpi:[10,16],object:[0,1,2,5,6,7,9,10,11,12],obligatori:[6,9],observ:[0,2,16],obtain:[0,6,9],offici:[15,16],often:[5,10],omit:16,on_failur:10,on_missing_vari:10,on_save_failur:10,on_unknown_typ:10,on_unused_sourc:[0,6],onc:[0,2,5,10,16],one:[1,2,4,5,6,7,9,10,16],ones:[0,2,16],onli:[0,1,2,4,5,7,10,16],oper:[1,2,5,16],opportun:10,optim:16,option:[0,6,7,10,11,12,14,16],order:[0,1,2,5,6,9,10,14,16],origin:[5,10,12],other:[0,1,10,16],otherwis:[6,11],our:[1,2,5,15,16],out:[15,16],outperform:10,output:[0,2,4,5,6,7,8,9,10,11,12,16],output_dir:[2,6,7,9,10,12],output_fil:10,output_nam:[5,6,10,11],output_root:7,outsid:2,over:[2,16],overal:0,overlap:16,overrid:[2,6,9,12],own:[2,5,10],packag:[13,16],param:[6,7,9,10,11],paramet:[0,1,5,6,7,9,10,11,12,16],parent:7,pars:[1,6,9,12],parse_arg:12,parse_fully_qualified_nam:12,parser:7,part:[0,5,16],pass:[0,2,4,5,6,7,9,10,11,12,16],path:[5,6,7,11,12],peek:2,perfectli:4,perform:[1,4,16],persist:5,petrbel:6,phase:[0,4],philosophi:3,pip:14,pipelin:1,place:2,placehold:16,pleas:[15,16],plot:1,plot_histogram:1,plugin:14,point:[0,2,5],polit:10,posit:5,possibl:[0,1],pow:16,practic:1,precis:16,predict:[0,1,2,3,5,6,7,16],predict_stream:[1,16],predicted_anim:5,predicted_vari:[0,2],predit:2,prefix:7,prescrib:[6,9],present:[10,16],print:[1,7,10,12],print_statist:1,probabl:[0,1],proce:16,procedur:7,process:[0,2,3,4,5,6,10,11,16],produc:[0,10,16],producion:16,product:[0,5,16],profil:[2,4,10,12],progress:16,project:[15,16],proper:1,properli:[10,14],properti:5,provid:[0,1,2,5,6,10,13,16],pseudocod:[4,5],publish:5,purpos:[1,14,16],python:[1,2,6,9,12,14,16],qualifi:[0,6,11,12],quallifi:12,queri:[1,4],quit:[2,10],rabbit:1,rais:[2,6,10,12],random:[0,1,16],random_integ:16,randomli:16,rang:[1,5,16],rate:0,ratio:16,read_data_train:10,readi:[0,16],real:16,realli:[1,16],reason:[0,1],recogn:10,recognit:5,recommend:[0,1,14,16],redirect:12,reduc:0,reduce_mean:16,refer:[1,6,9,10,11,12,15],reflect:12,regard:4,regardless:[2,10],regist:[0,1,4,7,10,16],registr:[0,1,16],regular:[1,3,16],relat:[1,16],remain:0,remov:6,renam:16,repositori:14,repr:[],repres:[1,2,16],requir:[0,1,6,9,10,16],required_min_valu:10,reset:10,reshap:16,resolut:0,respect:[2,10],respons:[4,16],rest:0,restor:[3,6,7,11,12],restore_fallback:[5,6,11],restore_from:[5,6,7,11],result:[0,2,6,10,11],resum:[1,5,7,16],resume_dir:7,resus:0,retriev:10,reus:0,reusabl:[15,16],rework:4,rewrit:0,rgb:1,right:4,root:7,rotat:[0,1],roughli:[2,7],row:10,rtype:[6,11],run:[3,4,6,7,10,11,14,16],run_predict:6,run_train:6,safe:16,sake:16,same:[0,7,12],save:[0,2,5,6,7,10,11,12,16],save_failure_act:10,save_model:10,savebest:[2,10],savebestmodel:10,saveeveri:10,saverhook:10,scalar:10,scenario:7,score:2,scratch:7,script:[1,7],search:[7,12],searchabl:12,second:[0,1,2,5,10,16],section:[0,1,3,4,6,7,11,15,16],see:[0,2,3,5,14,15,16],select:4,self:[1,2,10,16],self_num_sigint:10,sentenc:1,separ:[0,1,12,16],sequenc:[6,10],serv:0,set:[0,1,6,7,10],setup:14,sever:[1,13],shall:[2,5,7,10],shape:[1,16],share:2,should:[1,4,5,6,7,9,10,11,12,16],sigint:[10,16],siginthook:16,signal:[10,16],signum:10,similar:[4,5],similarli:16,simpl:[0,1,2,4,12,14,16],simpler:16,simpli:16,simplic:[1,16],sinc:[0,16],singl:[1,4,5,7,10,16],situat:0,size:[0,1,4,6],skip:0,skip_zeroth_epoch:[0,6],smart:[15,16],snippet:[0,5,16],some:[0,1,2,4,16],somebodi:5,sometim:1,sourc:[0,1,2,5,6,7,9,10,11,12,14,15,16],span:12,special:[0,2],specif:[0,4,5,6,7,11],specifi:[0,1,2,5,6,7,10,11,12,16],speed:15,split:[1,16],sq_err:16,squar:16,standard:[0,2,7,8,10,16],start:[2,4,7,10,12],stat:10,statist:[1,2,5,10,16],statshook:16,std:[10,16],stderr:[7,10],stdout:12,stem:10,step:[4,7,16],stop:[2,10,12,16],stop_aft:10,stopaft:[2,7,10],stopper:10,store:[2,5,6,7,10,11,16],str:[5,6,7,9,10,11,12],stream:[0,1,2,4,6,7,9,10,16],stream_nam:[2,6,9,10],strict:[],string:[0,1,6,7,9,12,16],strongli:0,structur:[0,2],sub:12,subclass:5,submodul:[11,12],subsequ:[2,10],successfulli:5,suffix:[6,10,11,16],suggest:[0,1,4],suit:0,sum:16,summar:10,suppli:2,support:[0,10],suppos:[0,1,16],suppress:0,sure:16,syntax:0,sys:[],system:2,t_co:[10,12],take:[1,2,6,9,10],taken:[10,16],target:[0,12],task:1,tell:16,tensor:1,tensorboard:[2,7],tensorboardhook:0,tensorflow:[5,7,14,16],termin:[0,2,4,7,10],test:[0,1,2,3,10,14,16],test_batch:5,test_i:16,test_stream:[1,5,16],test_x:16,text_param:7,than:[2,10],thank:10,thei:[0,1,16],them:[0,1,2,7,10,16],therefor:[0,4,16],thi:[0,1,2,3,4,5,6,7,9,10,11,12,14,16],thing:[4,16],third:16,those:[4,16],threshold:10,through:[4,6,10,16],time:[1,7,10,12],timer:12,todo:[1,4,5],togeth:[1,16],token:0,top:0,topic:16,total:16,track:2,tracker:15,train:[0,1,2,3,5,6,7,8,10,11,16],train_batch:5,train_by_stream:6,train_i:16,train_log:7,train_op:16,train_stream:[1,4,5,6,9,16],train_x:16,trainabl:[6,11],trainig:2,trainingtermin:[2,10],translat:2,tri:[6,11],trigger:[2,4,10],truth:0,tune:0,tupl:12,tutori:[2,5,15],two:[0,1,2,4,5],txt:[7,14],type:[0,1,6,7,9,10,11,12],typeerror:10,typic:1,unanot:0,unbias:16,under:[1,2,7,12,15],understand:16,uniqu:7,unit:[1,16],unknown:10,unknown_type_act:10,unrecogn:10,unsupport:10,unus:6,updat:[1,4,5,6,7,11],upon:2,usag:12,use:[0,1,2,4,6,11,14,16],used:[0,1,4,5,6,7,8,10,11,16],useful:[1,2,16],useless:1,user:[0,14],using:[0,16],usual:[2,4,5,6,10,11,16],util:[6,10,13,14],valid:[0,1,2,4,7,10,16],valid_stream:1,validate_config:7,valu:[0,2,4,10,12,16],valueerror:10,variabl:[0,2,5,10,16],variable_nam:10,variou:[0,1,6,8,16],vector:16,verb:10,veri:[0,4,16],verif:16,via:[1,10,14,15],visual:1,wai:[0,1],want:[1,16],warn:[0,6,7,10,12],wast:1,watch:[2,16],weak:[6,9,10,11,12],webpag:0,well:[0,1,16],were:10,what:[0,2,4,5,16],when:[0,1,4,5,6,7,10,11,12,14,16],whenc:[0,7],where:[2,4,5,10,16],wherein:[7,10],whether:[1,4,5,10],which:[0,1,2,4,5,6,7,9,10,11,12,14,15,16],whole:[0,1,2,4,5,16],width:[0,1],wise:1,wish:5,within:10,without:[1,5],work:2,workflow:5,world:16,would:[0,2,5,8,16],wrap:[12,16],wrapper:16,write:[2,10],write_csv:10,writecsv:[2,10],writetensorboard:[2,7],written:[2,6,9,10],xxx:10,y_hat:16,yaml:[0,1,6,7,8,9,12,16],yield:[1,16],you:[0,2,5,15,16],your:[2,4,15,16],zero:16,zeroth:6},titles:["Configuration","Dataset","Hooks","Advanced","Main Loop","Model","<code class=\"docutils literal\"><span class=\"pre\">cxflow</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow.cli</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow.constants</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow.datasets</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow.hooks</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow.models</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow.utils</span></code>","API Reference","Getting Started","cxflow","Tutorial"],titleterms:{"class":[6,9,10,11,12],"function":[7,12],The:1,Using:16,addit:1,advanc:3,after_batch:2,after_epoch:2,api:13,basedataset:1,cli:7,conclus:0,configur:[0,2,16],constant:8,contribut:15,cxflow:[6,7,8,9,10,11,12,15,16],data:1,dataset:[0,1,4,9,16],develop:14,event:2,except:10,get:14,hook:[0,2,4,10,16],infer:0,instal:14,integr:4,introduct:16,lazi:1,licens:15,lifecycl:4,loop:[0,4,16],main:[0,4,16],method:1,model:[0,4,5,11,16],philosophi:1,predict:4,process:1,refer:13,regular:2,requir:14,restor:5,run:5,start:14,submodul:6,support:15,task:16,test:4,train:4,tutori:16,util:12,variabl:[6,8,12]}})