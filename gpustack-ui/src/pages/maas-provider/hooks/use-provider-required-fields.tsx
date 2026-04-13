import useAppUtils from '@/hooks/use-app-utils';
import { useIntl } from '@umijs/max';
import { ProviderEnum } from '../config/providers';
import { RequiredFields } from '../config/types';

const useProviderRequiredFields = () => {
  const intl = useIntl();
  const { getRuleMessage } = useAppUtils();

  const providerRequiredFieldsMap: Record<string, RequiredFields[]> = {
    [ProviderEnum.OPENAI]: [
      {
        type: 'Input',
        name: 'openaiCustomUrl',
        placeholder: 'http://<your-inference-server>/v1',
        required: false,
        label: {
          text: 'providers.form.custombeckendUrl',
          locale: true
        }
      }
    ],
    [ProviderEnum.AZURE]: [
      {
        type: 'Input',
        name: 'azureServiceUrl',
        required: true,
        label: {
          text: 'providers.form.azureServiceUrl',
          locale: true
        },
        rules: [
          {
            required: true,
            message: getRuleMessage('input', 'providers.form.azureServiceUrl')
          }
        ]
      }
    ],
    [ProviderEnum.OLLAMA]: [
      {
        type: 'Input',
        name: 'ollamaServerHost',
        required: true,
        label: {
          text: 'providers.form.ollamaServerHost',
          locale: true
        },
        rules: [
          {
            required: true,
            message: getRuleMessage('input', 'providers.form.ollamaServerHost')
          }
        ]
      },
      {
        type: 'Input',
        name: 'ollamaServerPort',
        required: true,
        label: {
          text: 'providers.form.ollamaServerPort',
          locale: true
        },
        rules: [
          {
            required: true,
            message: getRuleMessage('input', 'providers.form.ollamaServerPort')
          }
        ]
      }
    ],
    [ProviderEnum.HUNYUAN]: [
      {
        type: 'Input',
        name: 'hunyuanAuthId',
        required: true,
        label: {
          text: 'providers.form.hunyuanAuthId',
          locale: true
        },
        rules: [
          {
            required: true,
            message: getRuleMessage('input', 'providers.form.hunyuanAuthId')
          }
        ]
      },
      {
        type: 'Password',
        name: 'hunyuanAuthKey',
        required: true,
        label: {
          text: 'providers.form.hunyuanAuthKey',
          locale: true
        },
        rules: [
          {
            required: true,
            message: getRuleMessage('input', 'providers.form.hunyuanAuthKey')
          }
        ]
      }
    ],
    [ProviderEnum.CLOUDFLARE]: [
      {
        type: 'Input',
        name: 'cloudflareAccountId',
        required: true,
        label: {
          text: 'providers.form.cloudflareAccountId',
          locale: true
        },
        rules: [
          {
            required: true,
            message: getRuleMessage(
              'input',
              'providers.form.cloudflareAccountId'
            )
          }
        ]
      }
    ],
    [ProviderEnum.DEEPL]: [
      {
        type: 'Input',
        name: 'targetLang',
        required: true,
        label: {
          text: 'providers.form.targetLang',
          locale: true
        },
        rules: [
          {
            required: true,
            message: getRuleMessage('input', 'providers.form.targetLang')
          }
        ]
      }
    ],
    [ProviderEnum.BEDROCK]: [
      {
        type: 'Input',
        name: 'awsAccessKey',
        required: true,
        label: {
          text: 'AWS Access Key',
          locale: false
        },
        rules: [
          {
            required: true,
            message: getRuleMessage('input', 'AWS Access Key', false)
          }
        ]
      },
      {
        type: 'Password',
        name: 'awsSecretKey',
        required: true,
        label: {
          text: 'AWS Secret Key',
          locale: false
        },
        rules: [
          {
            required: true,
            message: getRuleMessage('input', 'AWS Secret Key', false)
          }
        ]
      },
      {
        type: 'Input',
        name: 'awsRegion',
        placeholder: intl.formatMessage(
          { id: 'common.help.eg' },
          { content: 'us-eest-1' }
        ),
        required: true,
        label: {
          text: 'providers.form.awsRegion',
          locale: true
        },
        rules: [
          {
            required: true,
            message: getRuleMessage('input', 'providers.form.awsRegion')
          }
        ]
      }
    ],
    [ProviderEnum.CUSTOM]: [
      {
        type: 'Select',
        name: 'supported_formats',
        required: false,
        label: {
          text: 'providers.form.supportedFormats',
          locale: true
        },
        description: {
          text: 'providers.form.supportedFormatsDesc',
          locale: true
        }
      },
      {
        type: 'Input',
        name: 'customBaseUrl',
        required: true,
        placeholder: 'https://api.example.com/v1',
        label: {
          text: 'providers.form.customBaseUrl',
          locale: true
        },
        rules: [
          {
            required: true,
            message: getRuleMessage('input', 'providers.form.customBaseUrl')
          }
        ]
      },
      {
        type: 'Input',
        name: 'customBaseUrlAnthropic',
        required: false,
        placeholder: 'https://anthropic.example.com',
        label: {
          text: 'providers.form.customBaseUrlAnthropic',
          locale: true
        },
        description: {
          text: 'providers.form.customBaseUrlAnthropicDesc',
          locale: true
        },
        dependsOn: {
          field: 'supported_formats',
          includes: ['openai', 'anthropic']
        }
      },
      {
        type: 'Select',
        name: 'auth_style',
        required: false,
        label: {
          text: 'providers.form.authStyle',
          locale: true
        }
      },
      {
        type: 'Select',
        name: 'auth_style_anthropic',
        required: false,
        label: {
          text: 'providers.form.authStyleAnthropic',
          locale: true
        },
        dependsOn: {
          field: 'supported_formats',
          includes: ['openai', 'anthropic']
        }
      },
      {
        type: 'Checkbox',
        name: 'strip_fields',
        label: {
          text: 'providers.form.stripFields',
          locale: true
        },
        description: {
          text: 'providers.form.stripFieldsDesc',
          locale: true
        }
      }
    ]
  };

  return providerRequiredFieldsMap;
};

export default useProviderRequiredFields;
