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
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

import cloud_img from '@site/static/img/docs/tutorials/cloud.png';
import container_img from '@site/static/img/docs/tutorials/container.png';


const WebDevCards = [
  {
    name: 'PeriFlow Cloud',
    png: cloud_img,
    url: {
      page: 'cloud/how_to_use_your_checkpoint',
    },
    description: 'Check out how to use PeriFlow as a managed service.'
  },
  {
    name: 'PeriFlow Container',
    png: container_img,
    url: {
      page: 'container/how_to_run_periflow_container',
    },
    description: 'Find out how to use PeriFlow in the form of a Docker image.'
  },
];

interface WebDevCardProps {
  name: string;
  png: string;
  url: {
    page: string;
  };
  description: string;
}

const WebDevCard: React.FC<WebDevCardProps> = props => {
  return (
    <div className="col col--6 margin-bottom--lg">
      <div className={styles.cardContainer}>
        <Link to={props.url.page} style={{textDecoration: "none"}}>
          <div className={styles.cardImage}>
            <img src={props.png} className={styles.featurePng}/>
          </div>
          <div className={styles.cardBody}>
            <h3>{props.name}</h3>
            <p>{props.description}</p>
          </div>
        </Link>
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
