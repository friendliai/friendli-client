/* eslint-disable global-require */

import React from 'react';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

import cloud_img from '@site/static/img/docs/tutorials/cloud.png';
import container_img from '@site/static/img/docs/tutorials/container.png';


const CardDatas = [
  {
    name: 'PeriFlow Cloud',
    png: cloud_img,
    url: {
      page: 'cloud/how_to_use_your_checkpoint',
    },
    description: 'Use PeriFlow as a managed cloud service.'
  },
  {
    name: 'PeriFlow Container',
    png: container_img,
    url: {
      page: 'container/how_to_run_periflow_container',
    },
    description: 'Use PeriFlow in the form of a Docker image.'
  },
];

interface CardProps {
  name: string;
  png: string;
  url: {
    page: string;
  };
  description: string;
}

const Card: React.FC<CardProps> = props => {
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

export function CardView(): JSX.Element {
  return (
    <div className="row">
      {CardDatas.map((special) => (
        <Card key={special.name} {...special} />
      ))}
    </div>
  );
}
