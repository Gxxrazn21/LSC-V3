// Design tokens Material 3 para Seña Android
// Paleta basada en M3 con acentos colombianos (verde LSC como primario para diferenciarla)

const SENA_M3_THEMES = {
  default: {
    name: 'Estándar',
    bg: '#F5F2ED',           // surface container lowest
    surface: '#FFFFFF',
    surfaceContainer: '#ECE6DD',
    surfaceContainerHigh: '#E6E0D6',
    surfaceVariant: '#E5DFCE',
    text: '#1C1B1F',         // on-surface
    textMuted: '#49454F',    // on-surface-variant
    textSubtle: '#79747E',
    primary: '#1A56DB',      // azul colombiano
    onPrimary: '#FFFFFF',
    primaryContainer: '#DBE5FF',
    onPrimaryContainer: '#001A41',
    secondary: '#B8860B',    // amarillo colombiano oscurecido
    secondaryContainer: '#FFE6A8',
    onSecondaryContainer: '#261900',
    tertiary: '#006D5E',
    tertiaryContainer: '#A3F2DE',
    error: '#B3261E',
    errorContainer: '#F9DEDC',
    outline: '#79747E',
    outlineVariant: '#CAC4D0',
    avatarFrom: '#1A56DB',
    avatarTo: '#4E86DF',
  },
  daltonic: {
    name: 'Daltónico',
    bg: '#F4F2EC',
    surface: '#FFFFFF',
    surfaceContainer: '#ECE6DC',
    surfaceContainerHigh: '#E5DFD3',
    surfaceVariant: '#E3DECB',
    text: '#17171A',
    textMuted: '#434048',
    textSubtle: '#6E6A74',
    primary: '#0A44B2',
    onPrimary: '#FFFFFF',
    primaryContainer: '#D0DEF8',
    onPrimaryContainer: '#00153A',
    secondary: '#C68A00',
    secondaryContainer: '#FFE19C',
    onSecondaryContainer: '#221500',
    tertiary: '#0A44B2',
    tertiaryContainer: '#D0DEF8',
    error: '#0A44B2',
    errorContainer: '#D0DEF8',
    outline: '#6E6A74',
    outlineVariant: '#C5BFC8',
    avatarFrom: '#0A44B2',
    avatarTo: '#4578CF',
  },
  dark: {
    name: 'Oscuro',
    bg: '#121214',
    surface: '#1E1E22',
    surfaceContainer: '#26262B',
    surfaceContainerHigh: '#2E2E34',
    surfaceVariant: '#3A3A3F',
    text: '#E6E1E5',
    textMuted: '#CAC4D0',
    textSubtle: '#938F99',
    primary: '#AEC6FF',
    onPrimary: '#002E6A',
    primaryContainer: '#0041A0',
    onPrimaryContainer: '#DBE5FF',
    secondary: '#F0C75C',
    secondaryContainer: '#5A4200',
    onSecondaryContainer: '#FFE6A8',
    tertiary: '#88D6C2',
    tertiaryContainer: '#005045',
    error: '#F2B8B5',
    errorContainer: '#8C1D18',
    outline: '#938F99',
    outlineVariant: '#49454F',
    avatarFrom: '#3A6AD9',
    avatarTo: '#6A92E6',
  },
};

// Icono Material — outlined style (Android)
const MdIcon = ({ name, size = 24, color = 'currentColor', stroke = 2 }) => {
  const p = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: color, strokeWidth: stroke, strokeLinecap: 'round', strokeLinejoin: 'round' };
  switch (name) {
    case 'hand':
      return <svg {...p}><path d="M7 11V6a2 2 0 014 0v5"/><path d="M11 11V4a2 2 0 014 0v7"/><path d="M15 11V6a2 2 0 014 0v8a7 7 0 01-7 7h-1a7 7 0 01-7-7v-1a2 2 0 014 0"/></svg>;
    case 'sign':
      return <svg {...p}><path d="M4 13v2a6 6 0 006 6h3a7 7 0 007-7V9a2 2 0 10-4 0v3"/><path d="M16 12V5a2 2 0 10-4 0v5"/><path d="M12 10V4a2 2 0 10-4 0v7"/><path d="M8 11V7a2 2 0 10-4 0v6"/></svg>;
    case 'camera':
      return <svg {...p}><path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"/><circle cx="12" cy="13" r="4"/></svg>;
    case 'mic':
      return <svg {...p}><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M19 10v2a7 7 0 01-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>;
    case 'avatar':
      return <svg {...p}><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0116 0"/></svg>;
    case 'chat':
      return <svg {...p}><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>;
    case 'book':
      return <svg {...p}><path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/></svg>;
    case 'school':
      return <svg {...p}><path d="M22 10L12 4 2 10l10 6 10-6z"/><path d="M6 12v5c0 2 3 3 6 3s6-1 6-3v-5"/></svg>;
    case 'clock':
      return <svg {...p}><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>;
    case 'settings':
      return <svg {...p}><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 11-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 11-2.83-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 112.83-2.83l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 112.83 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>;
    case 'search':
      return <svg {...p}><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>;
    case 'swap':
      return <svg {...p}><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 014-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 01-4 4H3"/></svg>;
    case 'play':
      return <svg {...p} fill={color} stroke="none"><polygon points="6 4 20 12 6 20 6 4"/></svg>;
    case 'pause':
      return <svg {...p} fill={color} stroke="none"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>;
    case 'stop':
      return <svg {...p} fill={color} stroke="none"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>;
    case 'plus':
      return <svg {...p}><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>;
    case 'x':
      return <svg {...p}><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>;
    case 'volume':
      return <svg {...p}><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 010 7.07"/><path d="M19.07 4.93a10 10 0 010 14.14"/></svg>;
    case 'menu':
      return <svg {...p}><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>;
    case 'back':
      return <svg {...p}><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></svg>;
    case 'flip':
      return <svg {...p}><path d="M23 4v6h-6"/><path d="M1 20v-6h6"/><path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/></svg>;
    case 'check':
      return <svg {...p}><polyline points="20 6 9 17 4 12"/></svg>;
    case 'download':
      return <svg {...p}><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>;
    case 'offline':
      return <svg {...p}><line x1="1" y1="1" x2="23" y2="23"/><path d="M16.72 11.06A10.94 10.94 0 0119 12.55"/><path d="M5 12.55a10.94 10.94 0 015.17-2.39"/><path d="M10.71 5.05A16 16 0 0122.58 9"/><path d="M1.42 9a15.91 15.91 0 014.7-2.88"/><path d="M8.53 16.11a6 6 0 016.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/></svg>;
    case 'star':
      return <svg {...p}><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>;
    case 'bolt':
      return <svg {...p} fill={color} stroke="none"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>;
    case 'arrow-right':
      return <svg {...p}><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>;
    case 'arrow-down':
      return <svg {...p}><line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/></svg>;
    case 'arrow-up':
      return <svg {...p}><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>;
    case 'send':
      return <svg {...p}><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>;
    case 'info':
      return <svg {...p}><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>;
    case 'share':
      return <svg {...p}><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>;
    case 'dots':
      return <svg {...p}><circle cx="12" cy="12" r="1.5" fill={color}/><circle cx="12" cy="5" r="1.5" fill={color}/><circle cx="12" cy="19" r="1.5" fill={color}/></svg>;
    case 'filter':
      return <svg {...p}><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>;
    case 'notification':
      return <svg {...p}><path d="M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 01-3.46 0"/></svg>;
    case 'vibration':
      return <svg {...p}><rect x="8" y="5" width="8" height="14" rx="1"/><line x1="3" y1="9" x2="3" y2="15"/><line x1="21" y1="9" x2="21" y2="15"/></svg>;
    case 'eye':
      return <svg {...p}><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>;
    case 'edit':
      return <svg {...p}><path d="M17 3a2.85 2.83 0 114 4L7.5 20.5 2 22l1.5-5.5L17 3z"/></svg>;
    case 'more':
      return <svg {...p}><circle cx="5" cy="12" r="1.5" fill={color}/><circle cx="12" cy="12" r="1.5" fill={color}/><circle cx="19" cy="12" r="1.5" fill={color}/></svg>;
    default:
      return null;
  }
};

Object.assign(window, { SENA_M3_THEMES, MdIcon });
