// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/**
 * Defines a section with overridable defaults
 * @param {string} section
 * @param {import('@docusaurus/plugin-content-docs').Options} options
 */
function defineSection(section, version = {}, options = {}) {
  return [
    '@docusaurus/plugin-content-docs',
    /** @type {import('@docusaurus/plugin-content-docs').Options} */
    ({
      path: `docs/${section}`,
      routeBasePath: section,
      id: section,
      sidebarPath: require.resolve('./sidebars.js'),
      editUrl: 'https://github.com/friendliai/periflow-client/tree/main/',
      versions: version && {
        current: {
          label: version.label,
        },
      },
      ...options,
    }),
  ];
}

const latestVersions = {
  'cli': '0.1.3',
  'sdk': '0.1.1',
};

const SECTIONS = [
  defineSection('cli', {
    label: latestVersions['cli'],
  }),
  defineSection('sdk', {
    label: latestVersions['sdk'],
  }),
];

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'PeriFlow Documentation',
  tagline: 'Large-scale AI at ease',
  url: 'https://docs.periflow.ai',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'FriendliAI', // Usually your GitHub org/user name.
  projectName: 'docs', // Usually your repo name.

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          routeBasePath: '/',
          path: 'docs/home',
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/friendliai/periflow-client/tree/main/',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  plugins: [...SECTIONS],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/logo.svg',
      docs: {
        sidebar: {
          hideable: true,
        }
      },
      navbar: {
        title: '',
        logo: {
          alt: 'PeriFlow Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            label: 'CLI',
            to: 'cli/intro',
            position: 'left',
          },
          {
            label: 'Python SDK',
            to: 'sdk/intro',
            position: 'left',
          },
          {
            href: 'https://github.com/friendliai/periflow-client/',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'CLI',
                to: 'cli/intro',
              },
              {
                label: 'Python SDK',
                to: 'sdk/intro',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'PeriFlow Discuss',
                href: 'https://discuss.friendli.ai/',
              },
              {
                label: 'Twitter',
                href: 'https://twitter.com/friendliai',
              },
              {
                label: 'LinkedIn',
                href: 'https://www.linkedin.com/company/friendliai',
              },
              {
                label: 'Facebook',
                href: 'https://www.facebook.com/FriendliAI'
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'Blog',
                href: 'https://medium.com/friendliai',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/friendliai',
              },
            ],
          },
          {
            title: 'Legal Agreements',
            items: [
              {
                label: 'Privacy Policy',
                href: 'https://periflow.ai/policy',
              },
              {
                label: 'Terms of Use',
                href: 'https://periflow.ai/terms',
              },
              {
                label: 'Service Level Agreement',
                href: 'https://periflow.ai/service',
              }
            ]
          }
        ],
        copyright: `Copyright © ${new Date().getFullYear()} FriendlAI, Inc.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
};

module.exports = config;
