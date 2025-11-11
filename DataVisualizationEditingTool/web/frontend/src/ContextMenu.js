import React from 'react';
import './ContextMenu.css';

const ContextMenu = ({ x, y, visible, onSelect, items }) => {
  if (!visible) {
    return null;
  }

  return (
    <div className="context-menu" style={{ top: y, left: x }}>
      <ul>
        {items.map(item => (
          <li key={item.label} onClick={() => onSelect(item.action)}>
            {item.label}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default ContextMenu;
