t0_train_datasets = ['commonsense_qa', 'dream', 'quail', "quartz", 'social_i_qa', 'wiqa', 'cosmos_qa', 'qasc', 'quarel', 'sciq', 'wiki_hop', 
'adversarial_qa_dbert', 'adversarial_qa_dbidaf', 'adversarial_qa_droberta', 'quoref', 'duorc_ParaphraseRC', 'duorc_SelfRC', 'ropes', 'wiki_qa', 'common_gen', 'wiki_bio',
'app_reviews', 'amazon_polarity', 'imdb', 'rotten_tomatoes', 'gigaword', 'cnn_dailymail', 'multi_news', 'samsum',
'xsum', 'ag_news', 'dbpedia_14', 'trec', 'paws_labeled_final', 'glue_mrpc', 'glue_qqp', 'yelp_review_full', 'kilt_tasks_hotpotqa']

continual_train = {
  'gigaword': {
      'train': [
        ('custom', 'constrain_start+make_a_title'),
        ('custom', 'constrain_contain+make_a_title'),
        ('custom', 'constrain_end+make_a_title'),
      ],
  },
  'wiki_auto': {
      'train': [
        ('custom', 'simplification_1'),
      ]
  },
  'eli5': {
      'train_asks': [
        ('custom', 'generate_a_question_1'),
      ]
  },
  'covidfact': {
      'train': [
        ('t0_template', "__RANDOM__"),
      ]
  },
  'haiku': {
      'train': [
        ('custom', "do_nothing"),
      ]
  },
}

continual_test = {
  'gigaword': {
      'test': [
        ('t0_template', 'make_a_title'),
        ('custom', 'constrain_start+make_a_title'),
        ('custom', 'constrain_contain+make_a_title'),
        ('custom', 'constrain_end+make_a_title'),
        ('t0_template', 'write_its_sentence'),
        ('custom', 'constrain_start+write_its_sentence'),
        ('custom', 'constrain_contain+write_its_sentence'),
        ('custom', 'constrain_end+write_its_sentence'),
      ]
  },

  'wiki_auto': {
      'test': [
       ('custom', 'simplification_1'),
       ('custom', 'simplification_2'),
      ] 
  },
  'eli5': {
      'test_asks': [
        ('custom', 'generate_a_question_1'),
      ]
  },
  'covidfact': {
      'test': [
        ('t0_template', "__RANDOM__"),
      ]
  },
  
  'haiku': {
      'test': [
        ('custom', "do_nothing"),
      ]
  },

  'winogrande': { 'validation': [('t0_template', "fill in the blank")]},
  'rte': {'validation': [('t0_template', "can we infer")]},
  'wic': {'validation': [('t0_template', "same_sense")]},
  'copa': {'validation': [('t0_template', "choose")]}, 
  'hellaswag': {'validation': [('t0_template', "__RANDOM__")]}, 
  'anli': {'test_r1': [('t0_template', "__RANDOM__")]}, 
  'cb': {'validation': [('t0_template', "__RANDOM__")]}, 
  'wsc': {'validation': [('t0_template', "__RANDOM__")]}, 
  'story_cloze': {'validation': [('t0_template', "__RANDOM__")]}, 
}


list_config_reharsals = [
  {
      'name_exp': "gigaword", 
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'gigaword',
          'prompts': {
              'constrain_start+make_a_title': 33000,
              'constrain_contain+make_a_title': 33000,
              'constrain_end+make_a_title': 34000,
          }
      },
      'reharsal': {
          'list_datasets': t0_train_datasets
      }
  },

  {
      'name_exp': "wiki_auto", 
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'wiki_auto',
          'prompts': {
              'simplification_1': 100000,
          }
      },
      'reharsal': {
          'list_datasets': t0_train_datasets
      }
  },

  {
      'name_exp': "covidfact", 
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'covidfact',
          'prompts': {
              "__RANDOM__": 100000,
          }
      },
      'reharsal': {
          'list_datasets': t0_train_datasets
      }
  },

  {
      'name_exp': "eli5", 
      'new_dataset': {
          'eval_mode': 'train_asks',
          'name': 'eli5',
          'prompts': {
              'generate_a_question_1': 100000,
          }
      },
      'reharsal': {
          'list_datasets': t0_train_datasets
      }
  },
  
  {
      'name_exp': "haiku", 
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'haiku',
          'prompts': {
              'do_nothing': 100000,
          }
      },
      'reharsal': {
          'list_datasets': t0_train_datasets
      }
  },

]


list_config_reharsals_sequential = [
 {
      'name_exp': "sequencial.gigaword.from.wiki_auto", 
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'gigaword',
          'prompts': {
              'constrain_start+make_a_title': 33000,
              'constrain_contain+make_a_title': 33000,
              'constrain_end+make_a_title': 34000,
          }
      },
      'reharsal': {
          'list_datasets': t0_train_datasets + ['wiki_auto']
      }
  },
 
  {
      'name_exp': "sequencial.covidfact.from.wiki_auto->gigaword", 
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'covidfact',
          'prompts': {
              "__RANDOM__": 100000,
          }
      },
      'reharsal': {
          'list_datasets': t0_train_datasets + ['wiki_auto', 'gigaword']
      }
  },
 
  {
      'name_exp': "sequencial.haiku.from.wiki_auto->gigaword->covidfact", 
      'new_dataset': {
          'eval_mode': 'train',
          'name': 'haiku',
          'prompts': {
              "do_nothing": 100000,
          }
      },
      'reharsal': {
          'list_datasets': t0_train_datasets + ['wiki_auto', 'gigaword', 'covidfact']
      }
  },
 
 {
      'name_exp': "sequencial.eli5.from.wiki_auto->gigaword->covidfact->haiku", 
      'new_dataset': {
          'eval_mode': 'train_asks',
          'name': 'eli5',
          'prompts': {
              "generate_a_question_1": 100000,
          }
      },
      'reharsal': {
          'list_datasets': t0_train_datasets + ['wiki_auto', 'gigaword', 'covidfact', 'haiku']
      }
  },

]

evaluation_zero_shot  = {
  'gigaword': {
      'test': {
        'write_its_sentence': {
            'type': 't0_template',
            'choice': '',
            'metrics': ['rouge']
        },
        'constrain_start+write_its_sentence': {
            'type': 'custom',
            'choice': '',
            'metrics': ['rouge', 'constrain_accuracy']
        },
        'constrain_contain+write_its_sentence': {
            'type': 'custom',
            'choice': '',
            'metrics': ['rouge', 'constrain_accuracy']
        },
        'constrain_end+write_its_sentence': {
            'type': 'custom',
            'choice': '',
            'metrics': ['rouge', 'constrain_accuracy']
        },
      }
  },
  'wiki_auto': {
      'test': {
        'simplification_2': {
            'type': 'custom',
            'choice': '',
            'metrics': ['rouge', ]
        }
      },
  },
}


evaluation_new_tasks = {
  'gigaword': {
      'test': {
        'make_a_title': {
            'type': 't0_template',
            'choice': '',
            'metrics': ['rouge', 'bleu']
        },
        'constrain_start+make_a_title': {
            'type': 'custom',
            'choice': '',
            'metrics': ['rouge', 'bleu', 'constrain_start']
        },
        'constrain_contain+make_a_title': {
            'type': 'custom',
            'choice': '',
            'metrics': ['rouge', 'bleu', 'constrain_contain']
        },
        'constrain_end+make_a_title': {
            'type': 'custom',
            'choice': '',
            'metrics': ['rouge', 'bleu', 'constrain_end']
        },
      },
  },
  'wiki_auto': {
      'test': {
        'simplification_1': {
            'type': 'custom',
            'choice': '',
            'metrics': ['rouge', 'bleu', 'accuracy', 'sari']
        }
      }
  },
  'eli5': {
      'test_asks': {
        'generate_a_question_1': {
            'type': 'custom',
            'choice': '',
            'metrics': ['rouge', 'bleu']
        }
      }
  },
  'covidfact': {
      'test': {
        '__RANDOM__': {
            'type': 't0_template',
            'choice': '',
            'metrics': ['rouge', 'bleu', 'accuracy']
        }
      }
  },
  
  'haiku': {
      'test': {
        'do_nothing': {
            'type': 'custom',
            'choice': '',
            'metrics': ['rouge', 'bleu']
        }
      }
  },

}

evaluation_T0evalsets = {
  'winogrande': { 
      'validation': {
          "fill in the blank": {
            'type': 't0_template',
            'choice': 'option1_option2',
            'metrics': ['rouge', 'bleu', 'accuracy']
          }
      }
  },
  'rte': { 
      'validation': {
          "can we infer": {
            'type': 't0_template',
            'choice': 'yes_no',
            'metrics': ['rouge', 'bleu', 'accuracy']
          }
      }
  },
  'wic': { 
      'validation': {
          "same_sense": {
            'type': 't0_template',
            'choice': 'yes_no',
            'metrics': ['rouge', 'bleu', 'accuracy']
          }
      }
  },
  'copa': { 
    'validation': {
      "choose": {
        'type': 't0_template',
        'choice': 'option1_option2',
        'metrics': ['rouge', 'bleu', 'accuracy']
      }
    }
  },
  'hellaswag': { 
      'validation': {
          "__RANDOM__": {
            'type': 't0_template',
            'choice': '',
            'metrics': ['rouge', 'bleu', 'accuracy']
          }
      }
  },

  'anli': { 
      'test_r1': {
          "__RANDOM__": {
            'type': 't0_template',
            'choice': '',
            'metrics': ['rouge', 'bleu', 'accuracy']
          }
      }
  },

  'cb': { 
      'validation': {
          "__RANDOM__": {
            'type': 't0_template',
            'choice': '',
            'metrics': ['rouge', 'bleu', 'accuracy']
          }
      }
  },

  'wsc': { 
      'validation': {
          "__RANDOM__": {
            'type': 't0_template',
            'choice': '',
            'metrics': ['rouge', 'bleu', 'accuracy']
          }
      }
  },

  'story_cloze': { 
      'validation': {
          "__RANDOM__": {
            'type': 't0_template',
            'choice': '',
            'metrics': ['rouge', 'bleu', 'accuracy']
          }
      }
  },
}
