import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the draw button', () => {
  render(<App />);
  const buttonElement = screen.getByText(/Draw/i);
  expect(buttonElement).toBeInTheDocument();
});
