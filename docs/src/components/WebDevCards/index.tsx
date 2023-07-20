/*
Copyright (c) Facebook, Inc. and its affiliates.
 
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */

/* eslint-disable global-require */

import React from 'react';
import clsx from 'clsx';
import Translate from '@docusaurus/Translate';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

import get_started from '@site/static/img/docs/tutorial/get_started.png';
import hf_bert from '@site/static/img/docs/tutorial/hf_bert.png';
import hf_bart from '@site/static/img/docs/tutorial/hf_bart.png';
import pretrain_wiki from '@site/static/img/docs/tutorial/pretrain_wiki.png';
import pretrain_mt from '@site/static/img/docs/tutorial/pretrain_mt.png';
import resnet_cifar from '@site/static/img/docs/tutorial/resnet_cifar.png';


const WebDevCards = [
  // {
  //   name: 'Get started',
  //   Svg: require('@site/static/img/docs/tutorial/get_started.svg').default,
  //   url: {
  //     page: 'get_started',
  //   },
  //   description: (
  //     <div>
  //       Get started with PeriFlow by a basic PyTorch example.
  //     </div> 
  //   ),
  // },
  {
    name: 'Get started',
    png: get_started,
    url: {
      page: 'get_started',
    },
    description: (
      <div>
        Get started with PeriFlow by a basic PyTorch example.
      </div> 
    ),
  },
  {
    name: 'Fine-tuning BERT with Hugging Face',
    png: hf_bert,
    url: {
      page: 'hf_bert_ft',
    },
    description: (
      <div>
        Learn to fine-tune BERT with HuggingFace using PeriFlow.
      </div> 
    ),
  },
  {
    name: 'Fine-tuning BART with Hugging Face',
    png: hf_bart,
    url: {
      page: 'hf_bart_ft',
    },
    description: (
      <div>
        Learn to fine-tune BART with HuggingFace using Periflow.
      </div> 
    ),
  },
  {
    name: 'Pre-train GPT-2 with WikiText-2',
    png: pretrain_wiki,
    url: {
      page: 'pth_arlm',
    },
    description: (
      <div>
        Learn to pre-train GPT-2 with WikiText-2 using PeriFlow.
      </div> 
    ),
  },
  {
    name: 'Pre-train GPT with Megatron-LM',
    png: pretrain_mt,
    url: {
      page: 'megatron',
    },
    description: (
      <div>
        Learn to pre-train GPT with Megatron-LM using PeriFlow.
      </div> 
    ),
  },
  {
    name: 'Training ResNet with CIFAR',
    png: resnet_cifar,
    url: {
      page: 'cifar_10',
    },
    description: (
      <div>
        Learn to train ResNet on the CIFAR dataset using PeriFlow.
      </div> 
    ),
  },
];

// function WebDevCard({ name, Svg, url, description }) {
//   return (
//     <div className="col col--6 margin-bottom--lg">
//       <div className={clsx('card')}>
//         <div className={clsx('card__image')}>
//           <Link to={url.page}>
//             <Svg className={styles.featureSvg} role="img"/>
//           </Link>
//         </div>
//         <div className="card__body">
//           <h3>{name}</h3>
//           <p>{description}</p>
//         </div>
//         <div className="card__footer">
//           <div className="button-group button-group--block">
//             <Link className="button button--secondary" to={url.page}>
//               <Translate id="special.tryItButton">View Now!</Translate>
//             </Link>
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// }

function WebDevCard({ name, png, url, description }) {
  return (
    <div className="col col--6 margin-bottom--lg">
      <div className={clsx('card')}>
        <div className={clsx('card__image')}>
          <Link to={url.page}>
            <img src={png} className={styles.featurePng}/>
          </Link>
        </div>
        <div className="card__body">
          <h3>{name}</h3>
          <p>{description}</p>
        </div>
        <div className="card__footer">
          <div className="button-group button-group--block">
            <Link className="button button--secondary" to={url.page}>
              <Translate id="special.tryItButton">View Now!</Translate>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

export function WebDevCardsRow(): JSX.Element {
  return (
    <div className="row">
      {WebDevCards.map((special) => (
        <WebDevCard key={special.name} {...special} />
      ))}
    </div>
  );
}
