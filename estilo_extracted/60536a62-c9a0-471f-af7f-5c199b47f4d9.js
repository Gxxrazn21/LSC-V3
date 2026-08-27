// Pantallas Android 5-8: Diccionario, Aprender, Historial, Ajustes

const MdDictionaryScreen = ({ theme, goTo }) => {
  const [query, setQuery] = React.useState('');
  const [category, setCategory] = React.useState('Todos');
  const cats = ['Todos', 'Saludos', 'Familia', 'Números', 'Comida', 'Emociones', 'Salud'];
  const words = [
    { w: 'Hola', cat: 'Saludos', sign: 'hola' },
    { w: 'Gracias', cat: 'Saludos', sign: 'gracias' },
    { w: 'Mamá', cat: 'Familia', sign: 'familia' },
    { w: 'Papá', cat: 'Familia', sign: 'familia' },
    { w: 'Casa', cat: 'Familia', sign: 'casa' },
    { w: 'Sí', cat: 'Saludos', sign: 'si' },
    { w: 'No', cat: 'Saludos', sign: 'no' },
    { w: 'Hermano', cat: 'Familia', sign: 'familia' },
  ];
  const filtered = words.filter(w =>
    (category === 'Todos' || w.cat === category) &&
    w.w.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <div style={{ flex: 1, overflowY: 'auto', paddingBottom: 100, background: theme.bg }} className="md-scroll">
      <MdTopBar theme={theme} title="Diccionario" subtitle="1.240 señas LSC oficiales" large
        leading={<MdIconBtn icon="menu" color={theme.text}/>}
        trailing={<MdIconBtn icon="filter" color={theme.text}/>}
      />

      <div style={{ padding: '0 20px 0' }}>
        <div style={{
          background: theme.surfaceContainerHigh, borderRadius: 28,
          padding: '4px 4px 4px 16px', display: 'flex', alignItems: 'center', gap: 10,
        }}>
          <MdIcon name="search" size={20} color={theme.textMuted} stroke={2.2}/>
          <input value={query} onChange={e => setQuery(e.target.value)}
            placeholder="Buscar seña o palabra"
            style={{
              flex: 1, background: 'transparent', border: 'none', outline: 'none',
              fontFamily: 'inherit', fontSize: 15, color: theme.text, padding: '14px 0',
            }}/>
          <button style={{
            width: 44, height: 44, borderRadius: 99, border: 'none',
            background: theme.primaryContainer, cursor: 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <MdIcon name="camera" size={18} color={theme.onPrimaryContainer} stroke={2.2}/>
          </button>
        </div>
      </div>

      <div style={{ padding: '14px 0 4px' }}>
        <div style={{ display: 'flex', gap: 8, overflowX: 'auto', padding: '0 20px' }} className="md-scroll">
          {cats.map(c => (
            <MdChip key={c} theme={theme} label={c} selected={category === c} onClick={() => setCategory(c)}/>
          ))}
        </div>
      </div>

      {category === 'Todos' && !query && (
        <div style={{ padding: '12px 20px 4px' }}>
          <MdCard theme={theme} variant="filled" style={{
            padding: 0, overflow: 'hidden',
            background: `linear-gradient(135deg, ${theme.avatarFrom}, ${theme.avatarTo})`,
          }}>
            <div style={{ display: 'flex', gap: 0 }}>
              <div style={{ padding: '18px 0 18px 18px' }}>
                <MdSignerAvatar theme={theme} size={110} sign="hola"/>
              </div>
              <div style={{ flex: 1, padding: '20px 18px', color: '#fff' }}>
                <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: 1.2, opacity: 0.85 }}>SEÑA DEL DÍA</div>
                <div style={{ fontSize: 24, fontWeight: 500, marginTop: 4, letterSpacing: -0.2 }}>Bienvenido</div>
                <div style={{ fontSize: 12, opacity: 0.9, marginTop: 6, lineHeight: 1.4 }}>
                  Mano abierta, palma al frente, ligero movimiento hacia adelante.
                </div>
              </div>
            </div>
          </MdCard>
        </div>
      )}

      <div style={{ padding: '14px 20px 0' }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: theme.textMuted, letterSpacing: 0.8, marginBottom: 10 }}>
          {filtered.length} RESULTADOS
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          {filtered.map((w, i) => (
            <MdCard key={i} theme={theme} style={{ padding: 10, cursor: 'pointer' }}>
              <div style={{
                width: '100%', aspectRatio: '1', borderRadius: 12,
                background: `linear-gradient(160deg, ${theme.avatarFrom}, ${theme.avatarTo})`,
                position: 'relative', overflow: 'hidden',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <MdSignerAvatar theme={theme} size={110} sign={w.sign} animating={false}/>
                <div style={{
                  position: 'absolute', top: 6, right: 6,
                  width: 26, height: 26, borderRadius: 99,
                  background: 'rgba(0,0,0,0.35)', backdropFilter: 'blur(8px)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  <MdIcon name="play" size={10} color="#fff"/>
                </div>
              </div>
              <div style={{ marginTop: 10, display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <div style={{ fontSize: 15, fontWeight: 500, color: theme.text }}>{w.w}</div>
                <div style={{ fontSize: 11, color: theme.textMuted }}>{w.cat}</div>
              </div>
            </MdCard>
          ))}
        </div>
      </div>

      {/* FAB */}
      <div style={{ position: 'absolute', right: 20, bottom: 110, zIndex: 30 }}>
        <MdFab theme={theme} icon="plus" label="Sugerir seña"/>
      </div>
    </div>
  );
};

const MdLearnScreen = ({ theme, goTo }) => {
  const units = [
    { title: 'Saludos y cortesía', done: 8, total: 8, status: 'done' },
    { title: 'Me presento', done: 6, total: 6, status: 'done' },
    { title: 'La familia', done: 4, total: 8, status: 'active' },
    { title: 'Números del 1 al 20', done: 0, total: 10, status: 'locked' },
    { title: 'Comida y bebida', done: 0, total: 12, status: 'locked' },
  ];

  return (
    <div style={{ flex: 1, overflowY: 'auto', paddingBottom: 100, background: theme.bg }} className="md-scroll">
      <MdTopBar theme={theme} title="Aprender LSC" subtitle="Nivel intermedio · Día 14"
        leading={<MdIconBtn icon="menu" color={theme.text}/>}
      />

      {/* Racha */}
      <div style={{ padding: '4px 20px 0' }}>
        <MdCard theme={theme} variant="filled" style={{
          background: theme.text, color: theme.bg,
          padding: 18, display: 'flex', alignItems: 'center', gap: 14,
        }}>
          <div style={{
            width: 56, height: 56, borderRadius: 16,
            background: theme.secondary, color: theme.text,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontWeight: 700, fontSize: 24,
          }}>14</div>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 11, fontWeight: 600, opacity: 0.6, letterSpacing: 1 }}>RACHA ACTUAL</div>
            <div style={{ fontSize: 17, fontWeight: 500, marginTop: 2 }}>14 días seguidos</div>
            <div style={{ fontSize: 12, opacity: 0.7, marginTop: 2 }}>Practica hoy para continuar</div>
          </div>
          <div style={{ display: 'flex', gap: 4 }}>
            {['L','M','X','J','V','S','D'].map((d, i) => (
              <div key={i} style={{
                width: 18, height: 18, borderRadius: 99,
                background: i < 5 ? theme.secondary : 'rgba(255,255,255,0.15)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 9, fontWeight: 700,
                color: i < 5 ? theme.text : 'rgba(255,255,255,0.5)',
              }}>{d}</div>
            ))}
          </div>
        </MdCard>
      </div>

      {/* Continúa */}
      <div style={{ padding: '18px 20px 0' }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: theme.textMuted, letterSpacing: 0.8, marginBottom: 10 }}>
          CONTINÚA DONDE DEJASTE
        </div>
        <MdCard theme={theme} style={{
          padding: 0, overflow: 'hidden', background: theme.secondaryContainer,
        }}>
          <div style={{ padding: 18, display: 'flex', gap: 14, alignItems: 'center' }}>
            <MdSignerAvatar theme={theme} size={80} sign="familia"/>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: theme.onSecondaryContainer, opacity: 0.7, letterSpacing: 1 }}>UNIDAD 3 · LECCIÓN 5</div>
              <div style={{ fontSize: 17, fontWeight: 500, color: theme.onSecondaryContainer, marginTop: 2 }}>La familia</div>
              <div style={{ fontSize: 12, color: theme.onSecondaryContainer, opacity: 0.8, marginTop: 2 }}>Mamá, papá, hermanos</div>
            </div>
          </div>
          <div style={{ padding: '0 18px 14px' }}>
            <div style={{ height: 6, borderRadius: 99, background: 'rgba(0,0,0,0.1)', overflow: 'hidden' }}>
              <div style={{ width: '50%', height: '100%', background: theme.primary, borderRadius: 99 }}/>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8, fontSize: 11, fontWeight: 500, color: theme.onSecondaryContainer, opacity: 0.7 }}>
              <span>4 de 8 señas</span>
              <span>~6 min</span>
            </div>
          </div>
          <div style={{ padding: '0 14px 14px' }}>
            <MdButton theme={theme} icon="play" label="Continuar lección" variant="filled" fullWidth/>
          </div>
        </MdCard>
      </div>

      {/* Unidades */}
      <div style={{ padding: '18px 20px 0' }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: theme.textMuted, letterSpacing: 0.8, marginBottom: 10 }}>
          TODAS LAS UNIDADES
        </div>
        <div style={{ display: 'grid', gap: 10 }}>
          {units.map((u, i) => (
            <MdCard key={i} theme={theme} style={{
              padding: 14, display: 'flex', alignItems: 'center', gap: 12,
              opacity: u.status === 'locked' ? 0.55 : 1,
            }}>
              <div style={{
                width: 44, height: 44, borderRadius: 12,
                background: u.status === 'done' ? theme.tertiaryContainer
                         : u.status === 'active' ? theme.primaryContainer
                         : theme.surfaceVariant,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: u.status === 'done' ? theme.tertiary
                     : u.status === 'active' ? theme.onPrimaryContainer
                     : theme.textMuted,
                fontWeight: 600, fontSize: 15,
              }}>
                {u.status === 'done' ? <MdIcon name="check" size={18} color={theme.tertiary} stroke={3}/>
                 : u.status === 'locked' ? <MdIcon name="x" size={14} color={theme.textMuted}/>
                 : i + 1}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 15, fontWeight: 500, color: theme.text }}>{u.title}</div>
                <div style={{ fontSize: 12, color: theme.textMuted, marginTop: 2 }}>{u.done}/{u.total} señas</div>
              </div>
              {u.status === 'active' && <MdIcon name="arrow-right" size={18} color={theme.primary}/>}
              {u.status === 'done' && <MdIcon name="check" size={16} color={theme.tertiary} stroke={3}/>}
            </MdCard>
          ))}
        </div>
      </div>
    </div>
  );
};

const MdHistoryScreen = ({ theme, goTo }) => {
  const items = [
    { when: 'Hoy, 14:32', kind: 'conversation', title: 'Consulta médica', snippet: 'Hace tres días me duele el brazo…', duration: '8 min', participants: 2 },
    { when: 'Hoy, 09:12', kind: 'sign-to-text', title: 'Seña → Texto', snippet: 'Hola, mucho gusto. Me llamo Camila.', duration: '1 min' },
    { when: 'Ayer, 18:40', kind: 'text-to-sign', title: 'Texto → Seña', snippet: '¿Dónde está la estación de TransMilenio?', duration: '30 seg' },
    { when: 'Ayer, 11:08', kind: 'conversation', title: 'Reunión de trabajo', snippet: 'El reporte del trimestre está listo…', duration: '22 min', participants: 3 },
    { when: 'Lun 14 abr', kind: 'sign-to-text', title: 'Seña → Texto', snippet: 'Gracias por la clase de hoy.', duration: '45 seg' },
  ];
  const iconFor = k => k === 'conversation' ? 'chat' : k === 'sign-to-text' ? 'camera' : 'avatar';
  const colorBg = (k) => k === 'conversation' ? theme.primaryContainer
                      : k === 'sign-to-text' ? theme.secondaryContainer
                      : theme.tertiaryContainer;
  const colorFg = (k) => k === 'conversation' ? theme.onPrimaryContainer
                      : k === 'sign-to-text' ? theme.onSecondaryContainer
                      : theme.tertiary;

  return (
    <div style={{ flex: 1, overflowY: 'auto', paddingBottom: 100, background: theme.bg }} className="md-scroll">
      <MdTopBar theme={theme} title="Historial" subtitle="Traducciones guardadas" large
        leading={<MdIconBtn icon="menu" color={theme.text}/>}
        trailing={<MdIconBtn icon="search" color={theme.text}/>}
      />

      <div style={{ padding: '0 20px 14px', display: 'flex', gap: 8, overflowX: 'auto' }} className="md-scroll">
        {['Todo', 'Conversaciones', 'Señas', 'Favoritos'].map((f, i) => (
          <MdChip key={f} theme={theme} label={f} selected={i === 0}/>
        ))}
      </div>

      <div style={{ padding: '0 20px', display: 'grid', gap: 10 }}>
        {items.map((it, i) => (
          <MdCard key={i} theme={theme} style={{ padding: 14, display: 'flex', gap: 12 }}>
            <div style={{
              width: 44, height: 44, borderRadius: 12,
              background: colorBg(it.kind),
              display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
            }}>
              <MdIcon name={iconFor(it.kind)} size={20} color={colorFg(it.kind)} stroke={2.2}/>
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8 }}>
                <div style={{ fontSize: 15, fontWeight: 500, color: theme.text }}>{it.title}</div>
                <div style={{ fontSize: 11, color: theme.textSubtle, flexShrink: 0 }}>{it.when}</div>
              </div>
              <div style={{
                fontSize: 13, color: theme.textMuted, marginTop: 4, lineHeight: 1.35,
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>{it.snippet}</div>
              <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
                <div style={{
                  display: 'inline-flex', alignItems: 'center', gap: 4,
                  padding: '3px 8px', borderRadius: 6,
                  background: theme.surfaceVariant, color: theme.textMuted,
                  fontSize: 11, fontWeight: 500,
                }}>
                  <MdIcon name="clock" size={11} color={theme.textMuted} stroke={2.4}/>
                  {it.duration}
                </div>
                {it.participants && (
                  <div style={{
                    display: 'inline-flex', alignItems: 'center', gap: 4,
                    padding: '3px 8px', borderRadius: 6,
                    background: theme.surfaceVariant, color: theme.textMuted,
                    fontSize: 11, fontWeight: 500,
                  }}>
                    <MdIcon name="avatar" size={11} color={theme.textMuted} stroke={2.4}/>
                    {it.participants}
                  </div>
                )}
              </div>
            </div>
          </MdCard>
        ))}
      </div>
    </div>
  );
};

const MdSettingsScreen = ({ theme, themeKey, setThemeKey, fontScale, setFontScale, captions, setCaptions, offline, setOffline, vibration, setVibration, goTo }) => {
  const Switch = ({ on, onChange }) => (
    <button onClick={() => onChange(!on)} style={{
      width: 52, height: 32, borderRadius: 99, border: `2px solid ${on ? theme.primary : theme.outline}`,
      background: on ? theme.primary : 'transparent',
      position: 'relative', cursor: 'pointer', flexShrink: 0,
      transition: 'all 0.2s',
    }}>
      <div style={{
        position: 'absolute', top: '50%', left: on ? 24 : 4, transform: 'translateY(-50%)',
        width: on ? 20 : 14, height: on ? 20 : 14, borderRadius: 99,
        background: on ? '#fff' : theme.outline,
        transition: 'all 0.2s',
      }}/>
    </button>
  );

  const Row = ({ icon, iconBg, iconColor, title, sub, right, last }) => (
    <div style={{
      padding: '14px 16px', display: 'flex', alignItems: 'center', gap: 14,
      borderBottom: last ? 'none' : `1px solid ${theme.outlineVariant}33`,
    }}>
      <div style={{
        width: 40, height: 40, borderRadius: 99,
        background: iconBg || theme.primaryContainer,
        display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
      }}>
        <MdIcon name={icon} size={20} color={iconColor || theme.onPrimaryContainer} stroke={2.2}/>
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 15, fontWeight: 500, color: theme.text }}>{title}</div>
        {sub && <div style={{ fontSize: 12, color: theme.textMuted, marginTop: 2 }}>{sub}</div>}
      </div>
      {right}
    </div>
  );

  return (
    <div style={{ flex: 1, overflowY: 'auto', paddingBottom: 100, background: theme.bg }} className="md-scroll">
      <MdTopBar theme={theme} title="Ajustes" large
        leading={<MdIconBtn icon="back" color={theme.text} onClick={() => goTo('home')}/>}
      />

      {/* Perfil */}
      <div style={{ padding: '0 20px 0' }}>
        <MdCard theme={theme} style={{ padding: 16, display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{
            width: 56, height: 56, borderRadius: 99,
            background: `linear-gradient(135deg, ${theme.avatarFrom}, ${theme.avatarTo})`,
            color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 22, fontWeight: 500,
          }}>C</div>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 17, fontWeight: 500, color: theme.text }}>Camila Ospina</div>
            <div style={{ fontSize: 12, color: theme.textMuted, marginTop: 2 }}>Estudiante LSC · Bogotá</div>
          </div>
          <MdButton theme={theme} label="Editar" variant="tonal" icon="edit"/>
        </MdCard>
      </div>

      {/* Accesibilidad */}
      <div style={{ padding: '22px 20px 0' }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: theme.textMuted, letterSpacing: 0.8, marginBottom: 10, paddingLeft: 4 }}>
          ACCESIBILIDAD
        </div>
        <MdCard theme={theme} style={{ padding: 0, overflow: 'hidden' }}>
          <Row icon="eye" title="Modo daltónico"
            sub="Paleta azul + ámbar, evita rojo/verde"
            right={<Switch on={themeKey === 'daltonic'} onChange={v => setThemeKey(v ? 'daltonic' : 'default')}/>}/>
          <Row icon="chat" iconBg={theme.secondaryContainer} iconColor={theme.onSecondaryContainer}
            title="Subtítulos siempre visibles" sub="En conversaciones y videos"
            right={<Switch on={captions} onChange={setCaptions}/>}/>
          <Row icon="vibration" iconBg={theme.tertiaryContainer} iconColor={theme.tertiary}
            title="Vibración en detección" sub="Avisa cuando reconoce una seña"
            right={<Switch on={vibration} onChange={setVibration}/>}/>

          <div style={{ padding: '14px 16px', borderBottom: 'none' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 12 }}>
              <div style={{
                width: 40, height: 40, borderRadius: 99, background: theme.secondaryContainer,
                display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
              }}>
                <span style={{ fontWeight: 600, color: theme.onSecondaryContainer, fontSize: 16 }}>Aa</span>
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 15, fontWeight: 500, color: theme.text }}>Tamaño de texto</div>
                <div style={{ fontSize: 12, color: theme.textMuted }}>
                  {fontScale === 'normal' ? 'Normal' : fontScale === 'large' ? 'Grande' : 'Muy grande'}
                </div>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              {[['normal','Aa'],['large','Aa'],['xlarge','Aa']].map(([k, t], i) => (
                <button key={k} onClick={() => setFontScale(k)} style={{
                  flex: 1, padding: '10px', borderRadius: 12,
                  background: fontScale === k ? theme.primary : theme.surfaceVariant,
                  color: fontScale === k ? theme.onPrimary : theme.text,
                  border: 'none', cursor: 'pointer', fontFamily: 'inherit',
                  fontSize: 13 + i * 3, fontWeight: 500,
                }}>{t}</button>
              ))}
            </div>
          </div>
        </MdCard>
      </div>

      {/* Apariencia */}
      <div style={{ padding: '22px 20px 0' }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: theme.textMuted, letterSpacing: 0.8, marginBottom: 10, paddingLeft: 4 }}>
          APARIENCIA
        </div>
        <MdCard theme={theme} style={{ padding: 14 }}>
          <div style={{ display: 'flex', gap: 10 }}>
            {[
              { k: 'default', label: 'Claro', bg: '#F5F2ED', fg: '#1A56DB' },
              { k: 'daltonic', label: 'Daltónico', bg: '#F4F2EC', fg: '#0A44B2' },
              { k: 'dark', label: 'Oscuro', bg: '#121214', fg: '#AEC6FF' },
            ].map(t => (
              <button key={t.k} onClick={() => setThemeKey(t.k)} style={{
                flex: 1, padding: 10, borderRadius: 16,
                background: t.bg, border: themeKey === t.k ? `3px solid ${theme.primary}` : `1.5px solid ${theme.outlineVariant}`,
                cursor: 'pointer', fontFamily: 'inherit',
              }}>
                <div style={{
                  height: 48, borderRadius: 10,
                  background: `linear-gradient(135deg, ${t.fg}, ${t.fg}aa)`,
                  marginBottom: 8,
                }}/>
                <div style={{ fontSize: 12, fontWeight: 500, color: t.k === 'dark' ? '#fff' : '#1C1B1F' }}>
                  {t.label}
                </div>
              </button>
            ))}
          </div>
        </MdCard>
      </div>

      {/* Traducción */}
      <div style={{ padding: '22px 20px 0' }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: theme.textMuted, letterSpacing: 0.8, marginBottom: 10, paddingLeft: 4 }}>
          TRADUCCIÓN
        </div>
        <MdCard theme={theme} style={{ padding: 0, overflow: 'hidden' }}>
          <Row icon="offline" iconBg={theme.secondaryContainer} iconColor={theme.onSecondaryContainer}
            title="Modo sin conexión" sub="2 de 5 paquetes descargados"
            right={<Switch on={offline} onChange={setOffline}/>}/>
          <Row icon="download" title="Paquetes descargados"
            sub="Saludos, Familia · 180 MB"
            right={<MdIcon name="arrow-right" size={18} color={theme.textMuted}/>}/>
          <Row icon="avatar" iconBg={theme.tertiaryContainer} iconColor={theme.tertiary}
            title="Avatar firmante" sub="Paulina — voz colombiana"
            right={<MdIcon name="arrow-right" size={18} color={theme.textMuted}/>} last/>
        </MdCard>
      </div>

      {/* Privacidad */}
      <div style={{ padding: '22px 20px 0' }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: theme.textMuted, letterSpacing: 0.8, marginBottom: 10, paddingLeft: 4 }}>
          PRIVACIDAD
        </div>
        <MdCard theme={theme} style={{ padding: 0, overflow: 'hidden' }}>
          <Row icon="camera" title="Cámara"
            sub="Solo se procesa en tu dispositivo"
            right={<MdIcon name="check" size={18} color={theme.tertiary} stroke={3}/>}/>
          <Row icon="mic" title="Micrófono"
            right={<MdIcon name="arrow-right" size={18} color={theme.textMuted}/>} last/>
        </MdCard>
      </div>
    </div>
  );
};

Object.assign(window, { MdDictionaryScreen, MdLearnScreen, MdHistoryScreen, MdSettingsScreen });
