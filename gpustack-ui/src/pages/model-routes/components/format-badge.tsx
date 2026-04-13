import { Tag } from 'antd';

const FORMAT_STYLES: Record<string, { label: string; color: string }> = {
  openai: { label: 'O', color: '#818cf8' },
  anthropic: { label: 'A', color: '#f97316' },
};

interface FormatBadgeProps {
  supported_formats?: string[];
  style?: React.CSSProperties;
}

const FormatBadge: React.FC<FormatBadgeProps> = ({ supported_formats, style }) => {
  if (!supported_formats || supported_formats.length === 0) {
    return <Tag color="#818cf8" style={{ margin: 0, borderRadius: 8, fontSize: 11, lineHeight: '18px', ...style }}>O</Tag>;
  }

  if (supported_formats.length >= 2) {
    return <Tag color="#22c55e" style={{ margin: 0, borderRadius: 8, fontSize: 11, lineHeight: '18px', ...style }}>A+O</Tag>;
  }

  const fmt = supported_formats[0];
  const config = FORMAT_STYLES[fmt] || FORMAT_STYLES['openai'];
  return (
    <Tag color={config.color} style={{ margin: 0, borderRadius: 8, fontSize: 11, lineHeight: '18px', ...style }}>
      {config.label}
    </Tag>
  );
};

export default FormatBadge;
