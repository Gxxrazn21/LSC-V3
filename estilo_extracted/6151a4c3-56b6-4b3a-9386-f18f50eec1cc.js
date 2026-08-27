// Pantallas Android 1-4: Home, Seña→Texto, Voz→Seña, Conversación

const MdHomeScreen = ({ theme, goTo, user = 'Camila', isOffline }) => (
  <div style={{ flex: 1, overflowY: 'auto', paddingBottom: 100, background: theme.bg }} className="md-scroll">
    <MdTopBar theme={theme}
      leading={<MdIconBtn icon="menu" color={theme.text}/>}
      trailing={
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <MdIconBtn icon="notification" color={theme.text} size={20}/>
          <div style={{ marginRight: 8, marginLeft: 4,
            width: 36, height: 36, borderRadius: 99,
            background: theme.primary, color: theme.onPrimary,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontWeight: 500, fontSize: 15,
          }}>{user[0]}</div>
        </div>
      }
    />

    {/* Saludo */}
    <div style={{ padding: '4px 20px 0' }}>
      <div style={{ fontSize: 30, fontWeight: 400, color: theme.text, letterSpacing: -0.2, lineHeight: 1.15 }}>
        Hola, {user}
      </div>
      <div style={{ fontSize: 14, color: theme.textMuted, marginTop: 6 }}>
        ¿Qué quieres traducir hoy?
      </div>
      {isOffline && (
        <div style={{
          marginTop: 14, padding: '10px 14px', borderRadius: 12,
          background: theme.secondaryContainer, color: theme.onSecondaryContainer,
          display: 'flex', alignItems: 'center', gap: 10, fontSize: 13, fontWeight: 500,
        }}>
          <MdIcon name="offline" size={18} color={theme.onSecondaryContainer} stroke={2.2}/>
          Modo sin conexión · Traducción local disponible
        </div>
      )}
    </div>

    {/* Hero action M3-style (filled card tappable) */}
    <div style={{ padding: '20px 20px 0' }}>
      <button onClick={() => goTo('sign-to-text')} style={{
        width: '100%', textAlign: 'left', fontFamily: 'inherit',
        background: theme.primary, color: theme.onPrimary,
        border: 'none', borderRadius: 28, padding: 0, cursor: 'pointer',
        overflow: 'hidden', position: 'relative',
      }}>
        <div style={{ padding: '24px 22px 22px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 14 }}>
            <div style={{
              width: 52, height: 52, borderRadius: 16,
              background: 'rgba(255,255,255,0.22)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <MdIcon name="camera" size={26} color="#fff" stroke={2.2}/>
            </div>
            <div style={{ fontSize: 12, fontWeight: 600, opacity: 0.9, letterSpacing: 1.5 }}>
              INTERPRETAR SEÑAS
            </div>
          </div>
          <div style={{ fontSize: 24, fontWeight: 500, letterSpacing: -0.2, lineHeight: 1.2 }}>
            Abre la cámara y traduce LSC a texto en vivo
          </div>
          <div style={{
            marginTop: 18, display: 'inline-flex', alignItems: 'center', gap: 6,
            padding: '8px 14px', borderRadius: 99,
            background: 'rgba(255,255,255,0.22)', fontSize: 13, fontWeight: 500,
          }}>
            Empezar
            <MdIcon name="arrow-right" size={16} color="#fff" stroke={2.4}/>
          </div>
        </div>
      </button>
    </div>

    {/* Dos acciones secundarias */}
    <div style={{ padding: '12px 20px 0', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
      <button onClick={() => goTo('text-to-sign')} style={{
        background: theme.secondaryContainer, color: theme.onSecondaryContainer,
        border: 'none', borderRadius: 22, padding: 18, cursor: 'pointer',
        textAlign: 'left', fontFamily: 'inherit',
      }}>
        <MdIcon name="avatar" size={28} color={theme.onSecondaryContainer} stroke={2}/>
        <div style={{ fontSize: 16, fontWeight: 500, marginTop: 16, lineHeight: 1.25 }}>
          Voz o texto a seña
        </div>
        <div style={{ fontSize: 12, opacity: 0.7, marginTop: 4 }}>Avatar anima por ti</div>
      </button>

      <button onClick={() => goTo('conversation')} style={{
        background: theme.tertiaryContainer, color: theme.text,
        border: 'none', borderRadius: 22, padding: 18, cursor: 'pointer',
        textAlign: 'left', fontFamily: 'inherit', position: 'relative',
      }}>
        <MdIcon name="chat" size={28} color={theme.text} stroke={2}/>
        <div style={{ fontSize: 16, fontWeight: 500, marginTop: 16, lineHeight: 1.25 }}>
          Conversación
        </div>
        <div style={{ fontSize: 12, opacity: 0.7, marginTop: 4 }}>Cara a cara</div>
        <div style={{
          position: 'absolute', top: 14, right: 14,
          padding: '3px 8px', borderRadius: 6, background: theme.text, color: theme.bg,
          fontSize: 10, fontWeight: 700, letterSpacing: 0.5,
        }}>NUEVO</div>
      </button>
    </div>

    {/* Frases rápidas */}
    <div style={{ padding: '28px 0 0' }}>
      <div style={{ padding: '0 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 }}>
        <div style={{ fontSize: 16, fontWeight: 500, color: theme.text, letterSpacing: 0.1 }}>Frases rápidas</div>
        <button style={{ background: 'none', border: 'none', color: theme.primary, fontSize: 13, fontWeight: 500, cursor: 'pointer' }}>
          Ver todas
        </button>
      </div>
      <div style={{ display: 'flex', gap: 10, overflowX: 'auto', padding: '0 20px 4px' }} className="md-scroll">
        {[
          { t: 'Buenos días', icon: 'star' },
          { t: 'Gracias', icon: 'hand' },
          { t: '¿Cómo estás?', icon: 'chat' },
          { t: 'Necesito ayuda', icon: 'info' },
          { t: 'Me llamo…', icon: 'avatar' },
        ].map((q, i) => (
          <button key={i} onClick={() => goTo('text-to-sign')} style={{
            flexShrink: 0, background: theme.surfaceContainer,
            border: 'none', borderRadius: 16,
            padding: '14px 16px', cursor: 'pointer', fontFamily: 'inherit',
            display: 'flex', flexDirection: 'column', gap: 10, alignItems: 'flex-start',
            minWidth: 130,
          }}>
            <div style={{
              width: 36, height: 36, borderRadius: 10,
              background: theme.primaryContainer,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <MdIcon name={q.icon} size={18} color={theme.onPrimaryContainer} stroke={2.2}/>
            </div>
            <div style={{ fontSize: 13, fontWeight: 500, color: theme.text, textAlign: 'left' }}>{q.t}</div>
          </button>
        ))}
      </div>
    </div>

    {/* Lección actual */}
    <div style={{ padding: '24px 20px 0' }}>
      <MdCard theme={theme} onClick={() => goTo('learn')} style={{
        padding: 18, display: 'flex', alignItems: 'center', gap: 14,
      }}>
        <div style={{
          width: 52, height: 52, borderRadius: 14, background: theme.secondaryContainer,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <MdIcon name="bolt" size={26} color={theme.onSecondaryContainer}/>
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: theme.textMuted, letterSpacing: 1.2 }}>LECCIÓN 4 DE 12</div>
          <div style={{ fontSize: 16, fontWeight: 500, color: theme.text, marginTop: 2 }}>La familia</div>
          <div style={{ marginTop: 8, height: 4, borderRadius: 99, background: theme.surfaceVariant, overflow: 'hidden' }}>
            <div style={{ width: '50%', height: '100%', background: theme.primary, borderRadius: 99 }}/>
          </div>
        </div>
        <MdIcon name="play" size={22} color={theme.primary}/>
      </MdCard>
    </div>
  </div>
);

const MdSignToTextScreen = ({ theme, goTo }) => {
  const [recording, setRecording] = React.useState(true);
  const [detected, setDetected] = React.useState(['Hola', 'mucho', 'gusto']);
  React.useEffect(() => {
    if (!recording) return;
    const words = ['Hola','mucho','gusto','yo','me llamo','Camila','¿cómo','estás?'];
    let i = 3;
    const t = setInterval(() => {
      if (i >= words.length) return;
      setDetected(prev => [...prev, words[i]]);
      i++;
    }, 1800);
    return () => clearInterval(t);
  }, [recording]);

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: '#000' }}>
      <div style={{ padding: '12px 12px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <MdIconBtn icon="back" color="#fff" bg="rgba(255,255,255,0.15)"/>
        <div style={{
          padding: '8px 14px', borderRadius: 99,
          background: 'rgba(255,255,255,0.15)', backdropFilter: 'blur(12px)',
          display: 'flex', alignItems: 'center', gap: 8,
          color: '#fff', fontSize: 13, fontWeight: 500,
        }}>
          <div style={{ width: 8, height: 8, borderRadius: 99, background: '#FF5555', animation: 'mdPulse 1.4s infinite' }}/>
          Detectando señas
        </div>
        <MdIconBtn icon="flip" color="#fff" bg="rgba(255,255,255,0.15)"/>
      </div>

      <MdCameraPreview theme={theme} overlay={
        <>
          <div style={{
            position: 'absolute', top: '22%', left: '12%', right: '12%', bottom: '32%',
            border: '2px dashed rgba(255,255,255,0.35)',
            borderRadius: 20, pointerEvents: 'none',
          }}>
            <div style={{
              position: 'absolute', top: -11, left: 16,
              background: theme.primary, color: theme.onPrimary,
              padding: '3px 10px', borderRadius: 6,
              fontSize: 10, fontWeight: 600, letterSpacing: 0.5,
            }}>
              ENFOQUE LSC
            </div>
          </div>
          <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
            {[
              [135, 275], [125, 260], [145, 260], [130, 245], [140, 245],
              [165, 275], [155, 260], [175, 260], [160, 245], [170, 245],
            ].map(([x, y], i) => (
              <circle key={i} cx={`${(x/300)*100}%`} cy={`${(y/500)*100}%`} r="3" fill={theme.secondary}/>
            ))}
          </svg>
        </>
      }/>

      <div style={{
        background: theme.surface, borderTopLeftRadius: 28, borderTopRightRadius: 28,
        padding: '18px 20px 20px', maxHeight: '44%',
        display: 'flex', flexDirection: 'column', gap: 14,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 8,
            padding: '4px 10px', borderRadius: 6,
            background: theme.secondaryContainer, color: theme.onSecondaryContainer,
            fontSize: 11, fontWeight: 600, letterSpacing: 0.5,
          }}>
            <div style={{ width: 5, height: 5, borderRadius: 99, background: theme.secondary }}/>
            LSC · EN VIVO
          </div>
          <div style={{ display: 'flex', gap: 4 }}>
            <MdIconBtn icon="volume" color={theme.text} size={20}/>
            <MdIconBtn icon="share" color={theme.text} size={20}/>
          </div>
        </div>

        <div style={{
          flex: 1, overflowY: 'auto',
          fontSize: 22, fontWeight: 400, color: theme.text, lineHeight: 1.35, letterSpacing: -0.1,
        }} className="md-scroll">
          {detected.map((w, i) => (
            <span key={i} className="md-fade" style={{
              display: 'inline-block', marginRight: 8,
              background: i === detected.length - 1 ? theme.secondaryContainer : 'transparent',
              padding: i === detected.length - 1 ? '0 6px' : 0,
              borderRadius: 4,
              transition: 'background 0.3s',
            }}>{w}</span>
          ))}
          <span style={{
            display: 'inline-block', width: 2, height: 22, background: theme.primary,
            animation: 'mdPulse 1s infinite', marginLeft: 2, verticalAlign: 'middle',
          }}/>
        </div>

        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <MdButton theme={theme}
            icon={recording ? 'pause' : 'play'}
            label={recording ? 'Pausar' : 'Continuar'}
            variant="filled" fullWidth onClick={() => setRecording(!recording)}/>
          <MdButton theme={theme} icon="check" label="Listo" variant="tonal"/>
        </div>
      </div>
    </div>
  );
};

const MdTextToSignScreen = ({ theme, goTo }) => {
  const [input, setInput] = React.useState('Hola, mucho gusto. Me llamo Camila.');
  const [playing, setPlaying] = React.useState(true);
  const [currentWord, setCurrentWord] = React.useState(0);
  const words = input.replace(/[.,]/g, '').split(/\s+/).filter(Boolean);
  const signKeys = ['hola','gracias','si','no','familia','casa'];
  React.useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => setCurrentWord(w => (w + 1) % Math.max(words.length, 1)), 1400);
    return () => clearInterval(t);
  }, [playing, words.length]);

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: theme.bg }}>
      <MdTopBar theme={theme} title="Texto a seña"
        leading={<MdIconBtn icon="back" color={theme.text} onClick={() => goTo('home')}/>}
        trailing={<MdIconBtn icon="swap" color={theme.text}/>}
      />

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '8px 20px' }}>
        <MdSignerAvatar theme={theme} size={240} sign={signKeys[currentWord % signKeys.length]} animating={playing}/>
        <div style={{
          marginTop: 20, padding: '10px 22px',
          background: theme.surfaceContainer, borderRadius: 99,
          display: 'flex', alignItems: 'center', gap: 10,
        }}>
          <MdAudioBars color={theme.primary} active={playing} count={4} height={14}/>
          <span style={{ fontSize: 17, fontWeight: 500, color: theme.text, letterSpacing: 0 }}>
            {words[currentWord] || '—'}
          </span>
        </div>
      </div>

      {/* Timeline palabras */}
      <div style={{ padding: '0 20px 10px' }}>
        <div style={{ display: 'flex', gap: 6, overflowX: 'auto', paddingBottom: 4 }} className="md-scroll">
          {words.map((w, i) => (
            <MdChip key={i} theme={theme} label={w} selected={i === currentWord} onClick={() => setCurrentWord(i)}/>
          ))}
        </div>
      </div>

      {/* Playback */}
      <div style={{ padding: '6px 20px 14px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 16 }}>
        <button style={{
          width: 46, height: 46, borderRadius: 99,
          background: theme.surfaceContainer, border: 'none',
          cursor: 'pointer', fontSize: 11, fontWeight: 600, color: theme.text,
        }}>0.5×</button>
        <button onClick={() => setPlaying(!playing)} style={{
          width: 68, height: 68, borderRadius: 20,
          background: theme.primaryContainer, border: 'none', cursor: 'pointer',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <MdIcon name={playing ? 'pause' : 'play'} size={28} color={theme.onPrimaryContainer}/>
        </button>
        <button style={{
          width: 46, height: 46, borderRadius: 99,
          background: theme.surfaceContainer, border: 'none',
          cursor: 'pointer', fontSize: 11, fontWeight: 600, color: theme.text,
        }}>1.0×</button>
      </div>

      {/* Input */}
      <div style={{ background: theme.surfaceContainer, padding: '14px 16px 16px' }}>
        <div style={{
          background: theme.surface, borderRadius: 26,
          padding: '4px 4px 4px 16px', display: 'flex', alignItems: 'center', gap: 6,
        }}>
          <input value={input} onChange={e => setInput(e.target.value)}
            placeholder="Escribe o dicta…"
            style={{
              flex: 1, background: 'transparent', border: 'none', outline: 'none',
              fontFamily: 'inherit', fontSize: 15, color: theme.text, padding: '12px 0',
            }}/>
          <button style={{
            width: 44, height: 44, borderRadius: 99,
            background: 'transparent', border: 'none', cursor: 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <MdIcon name="mic" size={20} color={theme.text} stroke={2.2}/>
          </button>
          <button style={{
            width: 44, height: 44, borderRadius: 99,
            background: theme.primary, border: 'none', cursor: 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <MdIcon name="send" size={18} color={theme.onPrimary} stroke={2.2}/>
          </button>
        </div>
      </div>
    </div>
  );
};

const MdConversationScreen = ({ theme, goTo }) => {
  const [turn, setTurn] = React.useState('oyente');
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: theme.bg }}>
      <MdTopBar theme={theme} title="Conversación" subtitle="2 personas · 04:32"
        leading={<MdIconBtn icon="back" color={theme.text} onClick={() => goTo('home')}/>}
        trailing={<MdIconBtn icon="more" color={theme.text}/>}
      />

      {/* Panel oyente (invertido) */}
      <div style={{
        flex: 1, position: 'relative', margin: '4px 16px',
        borderRadius: 24, overflow: 'hidden',
        background: theme.secondaryContainer, color: theme.onSecondaryContainer,
        transform: 'rotate(180deg)',
      }}>
        <div style={{ position: 'absolute', inset: 0, padding: 20, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: 8,
              padding: '5px 12px', borderRadius: 8,
              background: 'rgba(0,0,0,0.08)',
              fontSize: 12, fontWeight: 600,
            }}>
              <MdIcon name="volume" size={13} color="currentColor" stroke={2.4}/>
              Oyente · Español
            </div>
            {turn === 'oyente' && (
              <div style={{
                display: 'inline-flex', alignItems: 'center', gap: 6,
                padding: '5px 10px', borderRadius: 99,
                background: theme.primary, color: theme.onPrimary,
                fontSize: 11, fontWeight: 600,
              }}>
                <MdAudioBars color={theme.onPrimary} active count={3} height={10}/>
                ESCUCHANDO
              </div>
            )}
          </div>
          <div className="md-fade">
            <div style={{ fontSize: 11, fontWeight: 600, opacity: 0.6, letterSpacing: 1, marginBottom: 6 }}>
              QUE ESCUCHÉ
            </div>
            <div style={{ fontSize: 22, fontWeight: 400, lineHeight: 1.3 }}>
              Claro, ¿desde cuándo empezaron los síntomas?
            </div>
          </div>
        </div>
      </div>

      {/* Cambio de turno */}
      <div style={{ display: 'flex', justifyContent: 'center', margin: '-14px 0', zIndex: 5, position: 'relative' }}>
        <button onClick={() => setTurn(turn === 'oyente' ? 'sordo' : 'oyente')} style={{
          width: 56, height: 56, borderRadius: 18,
          background: theme.text, color: theme.bg,
          border: `4px solid ${theme.bg}`,
          cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <MdIcon name="swap" size={22} color={theme.bg} stroke={2.4}/>
        </button>
      </div>

      {/* Panel sordo */}
      <div style={{
        flex: 1, position: 'relative', margin: '4px 16px 8px',
        borderRadius: 24, overflow: 'hidden',
        background: `linear-gradient(160deg, ${theme.avatarFrom}, ${theme.avatarTo})`,
      }}>
        <div style={{ position: 'absolute', inset: 0, padding: 20, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: 8,
              padding: '5px 12px', borderRadius: 8,
              background: 'rgba(255,255,255,0.25)', backdropFilter: 'blur(10px)',
              fontSize: 12, fontWeight: 600, color: '#fff',
            }}>
              <MdIcon name="hand" size={13} color="#fff" stroke={2.4}/>
              Persona sorda · LSC
            </div>
            <MdIconBtn icon="camera" color="#fff" bg="rgba(255,255,255,0.2)" size={18}/>
          </div>
          <div className="md-fade" style={{ color: '#fff' }}>
            <div style={{ fontSize: 11, fontWeight: 600, opacity: 0.85, letterSpacing: 1, marginBottom: 6 }}>
              TRADUCIENDO TU SEÑA
            </div>
            <div style={{ fontSize: 22, fontWeight: 400, lineHeight: 1.3 }}>
              Hace tres días me duele el brazo derecho
              <span style={{
                display: 'inline-block', width: 3, height: 22, background: theme.secondary,
                animation: 'mdPulse 1s infinite', marginLeft: 4, verticalAlign: 'middle',
              }}/>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom action bar */}
      <div style={{
        display: 'flex', justifyContent: 'center', gap: 14, padding: '12px 20px 16px',
        background: theme.surfaceContainer,
      }}>
        <MdIconBtn icon="mic" color={theme.text} bg={theme.surface}/>
        <button style={{
          height: 56, paddingInline: 24, borderRadius: 18,
          background: theme.error, color: '#fff', border: 'none', cursor: 'pointer',
          fontFamily: 'inherit', fontSize: 14, fontWeight: 600,
          display: 'flex', alignItems: 'center', gap: 8,
        }}>
          <MdIcon name="stop" size={18} color="#fff"/>
          Terminar
        </button>
        <MdIconBtn icon="volume" color={theme.text} bg={theme.surface}/>
      </div>
    </div>
  );
};

Object.assign(window, { MdHomeScreen, MdSignToTextScreen, MdTextToSignScreen, MdConversationScreen });
