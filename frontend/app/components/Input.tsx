type InputProps = {
  onSubmit: (value: string) => void;
};

export function Input({ onSubmit }: InputProps) {
  return (
    <button onClick={() => onSubmit("test")} type="button">
      Send
    </button>
  );
}
