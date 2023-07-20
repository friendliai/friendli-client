import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Easy to Use',
    Svg: require('@site/static/img/customize_data_ai.svg').default,
    description: (
      <>
        Don&apos;t be bothered to configure multiple clouds, regions, so many resources anymore.
        PeriFlow setup your team&apos;s MLOps journey from start to end with only few clicks!
      </>
    ),
  },
  {
    title: 'Fault Tolerant',
    Svg: require('@site/static/img/language_api.svg').default,
    description: (
      <>
        Even small failures are catastrophic in large ML models.
        PeriFlow orchestrates every training resource, recovers from faults, and automates scaling in/out resources.
      </>
    ),
  },
  {
    title: 'Serve Faster',
    Svg: require('@site/static/img/troubleshooting_ai.svg').default,
    description: (
      <>
        Servers should be both resource efficient and handle traffic well.
        PeriFlow automatically scales according to traffic and resource usage.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
