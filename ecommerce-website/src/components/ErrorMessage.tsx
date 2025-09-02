interface ErrorMessageProps {
  message: string;
}

function ErrorMessage({ message }: ErrorMessageProps) {
  return (
    <div className="error-message">
      <div className="error-content" data-testid="error-content">
        <p>{message}</p>
      </div>
    </div>
  );
}

// Export the component to be used in other parts of the application.
export default ErrorMessage;