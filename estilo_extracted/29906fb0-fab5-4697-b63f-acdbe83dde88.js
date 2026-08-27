// Material 3 shared components para Seña Android

// Avatar signer con estilo M3 (más plano)
const MdSignerAvatar = ({ theme, sign = 'hola', size = 220, animating = true }) => {
  const poses = {
    hola:     { lh: [60, 80],  rh: [140, 55], lr: -10, rr: 15 },
    gracias:  { lh: [90, 110], rh: [120, 90], lr: 0,   rr: -20 },
    si:       { lh: [70, 130], rh: [130, 130], lr: 0,  rr: 0 },
    no:       { lh: [85, 95],  rh: [125, 95], lr: -30, rr: 30 },
    familia:  { lh: [60, 100], rh: [140, 100], lr: 20, rr: -20 },
    casa:     { lh: [75, 70],  rh: [125, 70], lr: -45, rr: 45 },
  };
  const pose = poses[sign] || poses.hola;

  return (
    <div style={{
      width: size, height: size * 1.15, position: 'relative',
      borderRadius: 28, overflow: 'hidden',
      background: `linear-gradient(160deg, ${theme.avatarFrom} 0%, ${theme.avatarTo} 100%)`,
    }}>
      <div style={{
        position: 'absolute', inset: 0,
        background: 'radial-gradient(circle at 30% 20%, rgba(255,255,255,0.12), transparent 50%)',
      }}/>
      <svg viewBox="0 0 200 230" width="100%" height="100%" style={{ position: 'absolute', inset: 0 }}>
        <path d="M40 230 Q40 170 100 160 Q160 170 160 230 Z" fill="rgba(255,255,255,0.18)"/>
        <path d="M50 230 Q50 180 100 175 Q150 180 150 230 Z" fill="rgba(255,255,255,0.25)"/>
        <circle cx="100" cy="80" r="42" fill="rgba(255,255,255,0.95)"/>
        <circle cx="87" cy="78" r="3.5" fill={theme.text}/>
        <circle cx="113" cy="78" r="3.5" fill={theme.text}/>
        <path d="M90 95 Q100 101 110 95" stroke={theme.text} strokeWidth="2.5" fill="none" strokeLinecap="round"/>
        <g style={{
          transition: 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)',
          transformOrigin: '70px 145px',
          transform: animating ? `rotate(${pose.lr}deg)` : 'none',
        }}>
          <path d="M70 145 L65 175" stroke="rgba(255,255,255,0.95)" strokeWidth="16" strokeLinecap="round"/>
          <circle cx={pose.lh[0]} cy={pose.lh[1] + 60} r="13" fill="rgba(255,255,255,0.95)"/>
        </g>
        <g style={{
          transition: 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)',
          transformOrigin: '130px 145px',
          transform: animating ? `rotate(${pose.rr}deg)` : 'none',
        }}>
          <path d="M130 145 L135 175" stroke="rgba(255,255,255,0.95)" strokeWidth="16" strokeLinecap="round"/>
          <circle cx={pose.rh[0]} cy={pose.rh[1] + 60} r="13" fill="rgba(255,255,255,0.95)"/>
        </g>
      </svg>
      {animating && (
        <div style={{
          position: 'absolute', top: 12, right: 12,
          display: 'flex', alignItems: 'center', gap: 6,
          background: 'rgba(0,0,0,0.3)', backdropFilter: 'blur(8px)',
          padding: '5px 10px', borderRadius: 99,
          fontSize: 11, fontWeight: 500, color: '#fff',
        }}>
          <div style={{
            width: 6, height: 6, borderRadius: 99,
            background: '#FFE1A8',
            animation: 'mdPulse 1.4s ease-in-out infinite',
          }}/>
          Firmando
        </div>
      )}
    </div>
  );
};

// Card Material 3
const MdCard = ({ theme, children, style = {}, onClick, variant = 'filled' }) => {
  const bgMap = {
    filled: theme.surfaceContainer,
    elevated: theme.surface,
    outlined: theme.surface,
  };
  return (
    <div onClick={onClick} style={{
      background: bgMap[variant],
      borderRadius: 16,
      border: variant === 'outlined' ? `1px solid ${theme.outlineVariant}` : 'none',
      boxShadow: variant === 'elevated' ? '0 1px 3px rgba(0,0,0,0.12)' : 'none',
      cursor: onClick ? 'pointer' : 'default',
      ...style,
    }}>{children}</div>
  );
};

// FAB (botón flotante extendido)
const MdFab = ({ theme, icon, label, onClick, extended = true, size = 'md' }) => (
  <button onClick={onClick} style={{
    background: theme.primaryContainer, color: theme.onPrimaryContainer,
    border: 'none', cursor: 'pointer', fontFamily: 'inherit',
    borderRadius: extended ? 16 : (size === 'lg' ? 28 : 16),
    padding: extended ? '16px 20px' : (size === 'lg' ? '20px' : '16px'),
    display: 'flex', alignItems: 'center', gap: extended ? 12 : 0,
    fontSize: 14, fontWeight: 500, letterSpacing: 0.1,
    boxShadow: '0 3px 8px rgba(0,0,0,0.15), 0 1px 3px rgba(0,0,0,0.08)',
  }}>
    <MdIcon name={icon} size={size === 'lg' ? 26 : 22} color={theme.onPrimaryContainer} stroke={2}/>
    {extended && label}
  </button>
);

// Botón Material 3 (filled)
const MdButton = ({ theme, icon, label, onClick, variant = 'filled', fullWidth, size = 'md' }) => {
  const styles = {
    filled:  { bg: theme.primary, color: theme.onPrimary, border: 'none' },
    tonal:   { bg: theme.secondaryContainer, color: theme.onSecondaryContainer, border: 'none' },
    outlined:{ bg: 'transparent', color: theme.primary, border: `1px solid ${theme.outline}` },
    text:    { bg: 'transparent', color: theme.primary, border: 'none' },
  };
  const s = styles[variant];
  return (
    <button onClick={onClick} style={{
      background: s.bg, color: s.color, border: s.border,
      borderRadius: 99, padding: size === 'lg' ? '14px 28px' : '10px 20px',
      fontFamily: 'inherit', fontSize: 14, fontWeight: 500, letterSpacing: 0.1,
      cursor: 'pointer',
      width: fullWidth ? '100%' : undefined,
      display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 8,
    }}>
      {icon && <MdIcon name={icon} size={18} color={s.color} stroke={2.2}/>}
      {label}
    </button>
  );
};

// Chip Material 3
const MdChip = ({ theme, label, icon, selected, onClick, variant = 'assist' }) => {
  const bg = selected ? theme.secondaryContainer : 'transparent';
  const color = selected ? theme.onSecondaryContainer : theme.text;
  return (
    <button onClick={onClick} style={{
      background: bg, color,
      border: selected ? 'none' : `1px solid ${theme.outline}`,
      borderRadius: 8, padding: '6px 14px',
      fontFamily: 'inherit', fontSize: 13, fontWeight: 500, letterSpacing: 0.1,
      cursor: 'pointer', flexShrink: 0,
      display: 'inline-flex', alignItems: 'center', gap: 6,
    }}>
      {selected && <MdIcon name="check" size={14} color={color} stroke={2.5}/>}
      {icon && !selected && <MdIcon name={icon} size={14} color={color} stroke={2.2}/>}
      {label}
    </button>
  );
};

// Top app bar Material 3 (customizado para Seña)
const MdTopBar = ({ theme, title, subtitle, leading, trailing, large = false }) => (
  <div style={{ background: theme.bg, padding: large ? '4px 4px 12px' : '4px 4px 0' }}>
    <div style={{ height: 56, display: 'flex', alignItems: 'center', gap: 4 }}>
      {leading || <div style={{ width: 48, height: 48 }}/>}
      {!large && (
        <div style={{ flex: 1, paddingLeft: 4 }}>
          <div style={{ fontSize: 20, fontWeight: 500, color: theme.text, letterSpacing: 0.15 }}>
            {title}
          </div>
          {subtitle && (
            <div style={{ fontSize: 12, color: theme.textMuted, marginTop: 1 }}>{subtitle}</div>
          )}
        </div>
      )}
      {large && <div style={{ flex: 1 }}/>}
      {trailing}
    </div>
    {large && (
      <div style={{ padding: '12px 20px 16px' }}>
        <div style={{ fontSize: 30, fontWeight: 400, color: theme.text, letterSpacing: -0.2, lineHeight: 1.2 }}>
          {title}
        </div>
        {subtitle && (
          <div style={{ fontSize: 14, color: theme.textMuted, marginTop: 4 }}>{subtitle}</div>
        )}
      </div>
    )}
  </div>
);

// IconButton (48×48 touch target)
const MdIconBtn = ({ icon, onClick, color, size = 22, bg = 'transparent' }) => (
  <button onClick={onClick} style={{
    width: 48, height: 48, borderRadius: 99,
    background: bg, border: 'none', cursor: 'pointer',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  }}>
    <MdIcon name={icon} size={size} color={color} stroke={2.2}/>
  </button>
);

// Navigation bar inferior M3 — con "pill" en item activo
const MdNavBar = ({ theme, active, onNav }) => {
  const items = [
    { id: 'home',     icon: 'hand',     label: 'Inicio' },
    { id: 'dict',     icon: 'book',     label: 'Diccionario' },
    { id: 'learn',    icon: 'school',   label: 'Aprender' },
    { id: 'history',  icon: 'clock',    label: 'Historial' },
    { id: 'settings', icon: 'settings', label: 'Ajustes' },
  ];
  return (
    <div style={{
      position: 'absolute', bottom: 0, left: 0, right: 0, zIndex: 40,
      background: theme.surfaceContainer,
      paddingTop: 12, paddingBottom: 16,
      display: 'flex', justifyContent: 'space-around',
    }}>
      {items.map(it => {
        const isActive = active === it.id;
        return (
          <button key={it.id} onClick={() => onNav(it.id)} style={{
            background: 'none', border: 'none', cursor: 'pointer',
            display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4,
            padding: '0 4px', fontFamily: 'inherit', minWidth: 56,
          }}>
            <div style={{
              width: 64, height: 32, borderRadius: 99,
              background: isActive ? theme.secondaryContainer : 'transparent',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'background 0.2s',
            }}>
              <MdIcon name={it.icon} size={22} color={isActive ? theme.onSecondaryContainer : theme.textMuted} stroke={isActive ? 2.4 : 2}/>
            </div>
            <span style={{
              fontSize: 11, fontWeight: isActive ? 600 : 500,
              color: isActive ? theme.text : theme.textMuted,
              letterSpacing: 0.4,
            }}>{it.label}</span>
          </button>
        );
      })}
    </div>
  );
};

// Camera preview (reusable)
const MdCameraPreview = ({ theme, overlay, children }) => (
  <div style={{
    position: 'relative', flex: 1, overflow: 'hidden',
    background: '#0A0C14',
  }}>
    <div style={{
      position: 'absolute', inset: 0,
      background: `radial-gradient(ellipse at 50% 40%, #2A2E42 0%, #0A0C14 70%)`,
    }}/>
    <svg viewBox="0 0 300 500" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}>
      <ellipse cx="150" cy="180" rx="52" ry="62" fill="rgba(255,255,255,0.08)"/>
      <path d="M60 500 Q60 320 150 290 Q240 320 240 500 Z" fill="rgba(255,255,255,0.08)"/>
      <path d="M95 340 Q110 310 135 290" stroke="rgba(255,255,255,0.08)" strokeWidth="26" strokeLinecap="round" fill="none"/>
      <path d="M205 340 Q190 310 165 290" stroke="rgba(255,255,255,0.08)" strokeWidth="26" strokeLinecap="round" fill="none"/>
      <circle cx="135" cy="275" r="22" fill="rgba(255,255,255,0.12)"/>
      <circle cx="165" cy="275" r="22" fill="rgba(255,255,255,0.12)"/>
    </svg>
    <div style={{
      position: 'absolute', inset: 0,
      backgroundImage: `linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)`,
      backgroundSize: '30px 30px',
    }}/>
    {overlay}
    {children}
  </div>
);

// Audio bars
const MdAudioBars = ({ color, active = true, count = 5, height = 20 }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 3, height }}>
    {Array.from({ length: count }).map((_, i) => (
      <div key={i} style={{
        width: 3, height: '100%', borderRadius: 2,
        background: color,
        animation: active ? `mdBars ${0.7 + (i * 0.1)}s ease-in-out infinite` : 'none',
        animationDelay: `${i * 0.12}s`,
        transformOrigin: 'center',
        opacity: active ? 1 : 0.4,
      }}/>
    ))}
  </div>
);

const MdStyles = () => (
  <style>{`
    @keyframes mdPulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(1.4); }
    }
    @keyframes mdBars {
      0%, 100% { transform: scaleY(0.3); }
      50% { transform: scaleY(1); }
    }
    @keyframes mdFade {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes mdRipple {
      to { transform: scale(4); opacity: 0; }
    }
    .md-fade { animation: mdFade 0.4s ease-out both; }
    .md-scroll::-webkit-scrollbar { display: none; }
    .md-scroll { scrollbar-width: none; }
    button { font-family: inherit; }
    button:active { opacity: 0.85; }
  `}</style>
);

Object.assign(window, {
  MdSignerAvatar, MdCard, MdFab, MdButton, MdChip, MdTopBar,
  MdIconBtn, MdNavBar, MdCameraPreview, MdAudioBars, MdStyles,
});
