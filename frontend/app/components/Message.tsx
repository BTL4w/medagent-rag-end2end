type MessageProps = {
  text: string;
};

export function Message({ text }: MessageProps) {
  return <p>{text}</p>;
}
