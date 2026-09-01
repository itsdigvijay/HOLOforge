const slots = [1, 2, 3, 4, 5];

export function PowerSelector() {
  return (
    <nav className="power-selector" aria-label="Power selector placeholder">
      <p className="selector-label">POWER SELECT</p>
      <div className="power-slots">
        {slots.map((slot) => (
          <button
            className="power-slot"
            type="button"
            key={slot}
            disabled
            aria-label={`Power slot ${slot}, unavailable`}
          >
            <span>{slot.toString().padStart(2, '0')}</span>
          </button>
        ))}
      </div>
      <p className="selector-status">NO POWERS LOADED</p>
    </nav>
  );
}
