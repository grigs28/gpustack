import CheckboxField from '@/components/seal-form/checkbox-field';
import Password from '@/components/seal-form/password';
import SealInput from '@/components/seal-form/seal-input';
import SealSelect from '@/components/seal-form/seal-select';
import { useIntl } from '@umijs/max';
import { Form } from 'antd';
import { useFormContext } from '../config/form-context';
import { FormData } from '../config/types';

const SUPPORTED_FORMATS_OPTIONS = [
  { label: 'OpenAI', value: 'openai' },
  { label: 'Anthropic', value: 'anthropic' }
];

const AUTH_STYLE_OPTIONS = [
  { label: 'Auto', value: 'auto' },
  { label: 'Bearer', value: 'bearer' },
  { label: 'X-API-Key', value: 'x-api-key' }
];

const SELECT_OPTIONS_MAP: Record<string, Array<{ label: string; value: string }>> = {
  supported_formats: SUPPORTED_FORMATS_OPTIONS,
  auth_style: AUTH_STYLE_OPTIONS,
  auth_style_anthropic: AUTH_STYLE_OPTIONS
};

const ProviderConfigs = () => {
  const intl = useIntl();
  const form = Form.useFormInstance<FormData>();
  const { providerFields } = useFormContext();

  // Watch supported_formats for conditional rendering
  const supportedFormats = Form.useWatch(['config', 'supported_formats'], form);

  const renderLabel = (item: any) => {
    return item.label.locale
      ? intl.formatMessage({ id: item.label.text })
      : item.label.text;
  };

  const renderDescription = (item: any) => {
    return item.description
      ? item.description.locale
        ? intl.formatMessage({ id: item.description.text })
        : item.description.text
      : undefined;
  };

  const isFieldVisible = (item: any) => {
    if (!item.dependsOn) return true;
    const depValues = supportedFormats;
    if (!Array.isArray(depValues)) return false;
    return item.dependsOn.includes.every((v: string) => depValues.includes(v));
  };

  return (
    <>
      {providerFields && providerFields.length > 0
        ? providerFields?.map((item) => {
            if (!isFieldVisible(item)) return null;
            return (
              <Form.Item
                name={['config', item.name]}
                rules={item.rules}
                key={item.name}
                valuePropName={item.type === 'Checkbox' ? 'checked' : 'value'}
              >
                {item.type === 'Input' && (
                  <SealInput.Input
                    required={item.required}
                    description={renderDescription(item)}
                    label={renderLabel(item)}
                    placeholder={item.placeholder}
                  ></SealInput.Input>
                )}
                {item.type === 'Password' && (
                  <Password
                    required={item.required}
                    label={renderLabel(item)}
                    description={renderDescription(item)}
                    placeholder={item.placeholder}
                  ></Password>
                )}
                {item.type === 'Select' && (
                  <SealSelect
                    required={item.required}
                    label={renderLabel(item)}
                    description={renderDescription(item)}
                    options={SELECT_OPTIONS_MAP[item.name] || []}
                    mode={item.name === 'supported_formats' ? 'multiple' : undefined}
                    placeholder={item.placeholder}
                  ></SealSelect>
                )}
                {item.type === 'Checkbox' && (
                  <CheckboxField
                    label={renderLabel(item)}
                    description={renderDescription(item)}
                  />
                )}
              </Form.Item>
            );
          })
        : null}
    </>
  );
};

export default ProviderConfigs;
