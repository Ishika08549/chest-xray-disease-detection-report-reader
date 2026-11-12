import { useEffect, useState } from 'react'
import './App.css'

const API = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

const LABELS = ['Normal', 'Opacity', 'Cardiomegaly', 'Atelectasis']

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [predictions, setPredictions] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [selectedLabel, setSelectedLabel] = useState('Opacity')
  const [gradcamUrl, setGradcamUrl] = useState('')
  const [gradcamLoading, setGradcamLoading] = useState(false)
  const [imageId, setImageId] = useState('')
  const [report, setReport] = useState(null)
  const [reportLoading, setReportLoading] = useState(false)
  const [reportError, setReportError] = useState('')
  const [apiStatus, setApiStatus] = useState('Checking')

  useEffect(() => {
    fetch(`${API}/health`)
      .then((response) => {
        if (!response.ok) throw new Error('Backend unavailable')
        return response.json()
      })
      .then(() => setApiStatus('Connected'))
      .catch(() => setApiStatus('Disconnected'))
  }, [])

  function handleFileChange(event) {
    const selected = event.target.files?.[0]
    if (!selected) return

    if (!['image/png', 'image/jpeg', 'image/jpg'].includes(selected.type)) {
      setError('Please select a PNG or JPEG image.')
      return
    }

    if (selected.size > 10 * 1024 * 1024) {
      setError('Image size must be below 10 MB.')
      return
    }

    setFile(selected)
    setPreview(URL.createObjectURL(selected))
    setPredictions([])
    setGradcamUrl('')
    setReport(null)
    setError('')
    setReportError('')
  }

  async function handlePredict() {
    if (!file) {
      setError('Please upload a chest X-ray image first.')
      return
    }

    setLoading(true)
    setError('')
    setPredictions([])
    setGradcamUrl('')

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API}/predict`, {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail || 'Prediction failed.')
      }

      setPredictions(data.predictions || [])
    } catch (err) {
      setError(err.message || 'Unable to connect to the backend.')
    } finally {
      setLoading(false)
    }
  }

  async function handleGradcam() {
    if (!file) {
      setError('Upload an X-ray image first.')
      return
    }

    setGradcamLoading(true)
    setError('')

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('selected_label', selectedLabel)

      const response = await fetch(`${API}/gradcam`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Grad-CAM generation failed.')
      }

      const imageBlob = await response.blob()
      setGradcamUrl(URL.createObjectURL(imageBlob))
    } catch (err) {
      setError(err.message || 'Unable to generate Grad-CAM.')
    } finally {
      setGradcamLoading(false)
    }
  }

  async function handleReportSearch(event) {
    event.preventDefault()
    if (!imageId.trim()) {
      setReportError('Please enter an image ID.')
      return
    }

    setReportLoading(true)
    setReportError('')
    setReport(null)

    try {
      const response = await fetch(
        `${API}/report/${encodeURIComponent(imageId.trim())}`
      )
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Report not found.')
      }

      setReport(data.report)
    } catch (err) {
      setReportError(err.message || 'Unable to retrieve report.')
    } finally {
      setReportLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">✚</div>
          <div>
            <strong>Radiant AI</strong>
            <span>Research workspace</span>
          </div>
        </div>

        <div className="side-label">WORKSPACE</div>
        <a className="nav-link active" href="#overview">▦ Dashboard</a>
        <a className="nav-link" href="#upload">＋ X-ray analysis</a>
        <a className="nav-link" href="#gradcam">◉ Grad-CAM</a>
        <a className="nav-link" href="#reports">▤ Report reader</a>

        <div className="sidebar-bottom">
          <div className="api-indicator">
            <span className={apiStatus === 'Connected' ? 'status-dot online' : 'status-dot'} />
            <div>
              <strong>API status</strong>
              <span>{apiStatus}</span>
            </div>
          </div>
          <p>Educational research prototype</p>
        </div>
      </aside>

      <main className="main-content" id="overview">
        <header className="topbar">
          <div>
            <span className="eyebrow">CHEST X-RAY INTELLIGENCE</span>
            <h1>Analysis Dashboard</h1>
            <p>Explore model predictions and associated radiology reports.</p>
          </div>
          <div className="top-status">
            <span className="status-dot online" />
            Research mode
          </div>
        </header>

        <section className="welcome-panel">
          <div>
            <span className="eyebrow light">AI-POWERED RESEARCH TOOL</span>
            <h2>Chest X-Ray Disease Detection</h2>
            <p>
              Multi-label image classification, visual explanations and
              dataset-linked report retrieval in one workspace.
            </p>
          </div>
          <div className="welcome-symbol">✚</div>
        </section>

        <section className="section" id="upload">
          <div className="section-heading">
            <div>
              <span className="eyebrow">01 / IMAGE ANALYSIS</span>
              <h2>Upload chest X-ray</h2>
              <p>Choose a PNG or JPEG image up to 10 MB.</p>
            </div>
          </div>

          <div className="analysis-grid">
            <div className="panel upload-panel">
              <label className="upload-zone">
                <input
                  type="file"
                  accept="image/png,image/jpeg"
                  onChange={handleFileChange}
                />
                {preview ? (
                  <img className="xray-preview" src={preview} alt="Uploaded chest X-ray" />
                ) : (
                  <>
                    <div className="upload-icon">↑</div>
                    <strong>Choose an X-ray image</strong>
                    <span>Click to browse your files</span>
                    <small>PNG or JPEG · Maximum 10 MB</small>
                  </>
                )}
              </label>

              {file && <div className="file-name">Selected: {file.name}</div>}

              <button
                className="primary-button full-button"
                onClick={handlePredict}
                disabled={!file || loading}
              >
                {loading ? 'Analysing image...' : 'Run prediction →'}
              </button>
            </div>

            <div className="panel results-panel">
              <div className="panel-title">
                <div>
                  <span className="eyebrow">MODEL OUTPUT</span>
                  <h3>Prediction results</h3>
                </div>
                <span className="mini-tag">4 labels</span>
              </div>

              {predictions.length > 0 ? (
                <div className="prediction-list">
                  {predictions.map((item) => {
                    const percentage = Number(
                      item.probability_percent ??
                      (item.probability != null ? item.probability * 100 : 0)
                    )
                    return (
                      <div className="prediction-item" key={item.label}>
                        <div className="prediction-meta">
                          <strong>{item.label}</strong>
                          <span>{percentage.toFixed(2)}%</span>
                        </div>
                        <div className="progress-track">
                          <div
                            className="progress-fill"
                            style={{ width: `${Math.min(100, Math.max(0, percentage))}%` }}
                          />
                        </div>
                        <small>
                          {item.candidate_detected ? 'Above model threshold' : 'Below model threshold'}
                        </small>
                      </div>
                    )
                  })}
                </div>
              ) : (
                <div className="empty-state">
                  <div className="empty-icon">▤</div>
                  <strong>No predictions yet</strong>
                  <p>Upload an image and run the model to see its output here.</p>
                </div>
              )}
            </div>
          </div>
        </section>

        {error && <div className="error-message" role="alert">{error}</div>}

        <section className="section" id="gradcam">
          <div className="section-heading">
            <div>
              <span className="eyebrow">02 / MODEL EXPLAINABILITY</span>
              <h2>Grad-CAM visualization</h2>
              <p>Generate a heatmap for a selected model label.</p>
            </div>
          </div>

          <div className="panel gradcam-panel">
            <div className="gradcam-controls">
              <label htmlFor="label-select">Select prediction label</label>
              <select
                id="label-select"
                value={selectedLabel}
                onChange={(event) => setSelectedLabel(event.target.value)}
              >
                {LABELS.map((label) => <option key={label}>{label}</option>)}
              </select>
              <button
                className="primary-button"
                onClick={handleGradcam}
                disabled={!file || gradcamLoading}
              >
                {gradcamLoading ? 'Generating...' : 'Generate heatmap'}
              </button>
            </div>

            <div className="heatmap-area">
              {gradcamUrl ? (
                <img src={gradcamUrl} alt={`Grad-CAM heatmap for ${selectedLabel}`} />
              ) : (
                <div className="empty-state">
                  <div className="empty-icon">◉</div>
                  <strong>Heatmap will appear here</strong>
                  <p>Select a label and generate its visualization.</p>
                </div>
              )}
            </div>
            <p className="helper-text">
              Grad-CAM highlights image regions influencing the model output.
              It is an explanatory visualization, not proof of disease.
            </p>
          </div>
        </section>

        <section className="section" id="reports">
          <div className="section-heading">
            <div>
              <span className="eyebrow">03 / REPORT RETRIEVAL</span>
              <h2>Radiology report reader</h2>
              <p>Retrieve the original dataset report using its verified image ID.</p>
            </div>
          </div>

          <div className="panel report-panel">
            <form className="report-search" onSubmit={handleReportSearch}>
              <label htmlFor="image-id">Dataset image ID</label>
              <div className="search-row">
                <input
                  id="image-id"
                  value={imageId}
                  onChange={(event) => setImageId(event.target.value)}
                  placeholder="Example: CXR1_1_IM-0001-3001"
                />
                <button className="primary-button" type="submit" disabled={reportLoading}>
                  {reportLoading ? 'Searching...' : 'Find report'}
                </button>
              </div>
            </form>

            {reportError && <div className="error-message">{reportError}</div>}

            {report && (
              <div className="report-results">
                <div className="verified-banner">
                  <span>✓</span> Verified image-report mapping
                </div>
                <div className="report-meta">
                  <span>Image ID: <strong>{report.image_id}</strong></span>
                  <span>Report ID: <strong>{report.report_id}</strong></span>
                </div>
                <div className="report-block">
                  <h3>Indication</h3>
                  <p>{report.indication || 'Not available'}</p>
                </div>
                <div className="report-block">
                  <h3>Comparison</h3>
                  <p>{report.comparison || 'Not available'}</p>
                </div>
                <div className="report-block">
                  <h3>Findings</h3>
                  <p>{report.findings || 'Not available'}</p>
                </div>
                <div className="report-block">
                  <h3>Impression</h3>
                  <p>{report.impression || 'Not available'}</p>
                </div>
              </div>
            )}
          </div>
        </section>

        <footer className="disclaimer">
          <strong>Important educational disclaimer</strong>
          <p>
            This application is an educational research prototype, not a
            medical device. Model predictions and Grad-CAM visualizations
            are not medical diagnoses. Retrieved reports are original
            dataset records, not newly generated clinical interpretations.
          </p>
        </footer>
      </main>
    </div>
  )
}

export default App
