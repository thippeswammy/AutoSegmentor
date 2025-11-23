import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import PlotComponent from './components/Plot'
import Toolbar from './components/Toolbar'
import './App.css'

function App() {
  const [data, setData] = useState({ nodes: [], edges: [], file_names: [], D: 1.0 })
  const [loading, setLoading] = useState(true)
  const [status, setStatus] = useState("Initializing...")
  const [selectedPoints, setSelectedPoints] = useState([])

  // View Settings
  const [pointSize, setPointSize] = useState(10)
  const [smoothness, setSmoothness] = useState(1.0)

  const fetchState = async () => {
    try {
      const response = await axios.get('/api/state')
      setData(response.data)
      setLoading(false)
      setStatus("Ready")
    } catch (error) {
      console.error("Error fetching state:", error)
      setStatus("Error fetching state")
    }
  }

  useEffect(() => {
    const init = async () => {
      try {
        await axios.get('/api/init')
        fetchState()
      } catch (error) {
        console.error("Error initializing:", error)
        setStatus("Error initializing")
      }
    }
    init()
  }, [])

  const handleAddNode = async (x, y, laneId) => {
    try {
      await axios.post('/api/action/add_node', { x, y, lane_id: laneId })
      fetchState()
      setStatus("Node added")
    } catch (error) {
      console.error("Error adding node:", error)
    }
  }

  /**
   * Handles the addition of an edge between two selected points.
   *
   * This function checks if exactly two points are selected. If so, it sends a POST request to the API to add an edge
   * between the selected points using their IDs. Upon a successful request, it fetches the updated state and sets a
   * status message indicating that the edge has been added. In case of an error during the request, it logs the error
   * to the console.
   */
  const handleAddEdge = async () => {
    if (selectedPoints.length !== 2) return
    try {
      await axios.post('/api/action/add_edge', { from_id: selectedPoints[0], to_id: selectedPoints[1] })
      fetchState()
      setStatus("Edge added")
    } catch (error) {
      console.error("Error adding edge:", error)
    }
  }

  /**
   * Handles the deletion of selected points.
   *
   * This function checks if there are any selected points to delete. If there are, it sends a POST request to the API to delete the selected points. Upon successful deletion, it resets the selected points, fetches the updated state, and sets a status message. In case of an error during the deletion process, it logs the error to the console.
   */
  const handleDelete = async () => {
    if (selectedPoints.length === 0) return
    try {
      await axios.post('/api/action/delete', { point_ids: selectedPoints })
      setSelectedPoints([])
      fetchState()
      setStatus("Deleted selected points")
    } catch (error) {
      console.error("Error deleting:", error)
    }
  }

  /**
   * Handles the undo action by making an API call and updating the state.
   */
  const handleUndo = async () => {
    try {
      await axios.post('/api/action/undo')
      fetchState()
      setStatus("Undo performed")
    } catch (error) {
      console.error("Error undoing:", error)
    }
  }

  /**
   * Handles the redo action by making an API call and updating the state.
   */
  const handleRedo = async () => {
    try {
      await axios.post('/api/action/redo')
      fetchState()
      setStatus("Redo performed")
    } catch (error) {
      console.error("Error redoing:", error)
    }
  }

  const handleSave = async () => {
    try {
      setStatus("Saving...")
      const res = await axios.post('/api/save')
      setStatus(`Saved to ${res.data.path}`)
      setTimeout(() => setStatus("Ready"), 3000)
    } catch (error) {
      console.error("Error saving:", error)
      setStatus("Error saving")
    }
  }

  // Keyboard Shortcuts
  useEffect(() => {
    /**
     * Handles key down events for specific keyboard shortcuts.
     *
     * The function checks if the event target is an input element and ignores the event if so.
     * It processes key events for 'Delete', 'Backspace', 'Ctrl+Z', 'Ctrl+Y', and their variations,
     * invoking the appropriate handler functions (handleDelete, handleUndo, handleRedo)
     * while preventing default actions for certain combinations.
     *
     * @param e - The keyboard event object.
     */
    const handleKeyDown = (e) => {
      // Ignore if input is focused (though we don't have many inputs)
      if (e.target.tagName === 'INPUT') return;

      if (e.key === 'Delete' || e.key === 'Backspace') {
        handleDelete();
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        if (e.shiftKey) {
          handleRedo();
        } else {
          handleUndo();
        }
        e.preventDefault();
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'y') {
        handleRedo();
        e.preventDefault();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedPoints]); // Re-bind when selectedPoints changes to ensure handleDelete has current state? 
  // Actually, handleDelete uses selectedPoints from closure. 
  // Better to use useCallback or ref for selectedPoints if we don't want to re-bind often, 
  // but re-binding on selection change is acceptable here.

  return (
    <div className="app-container">
      <div className="header">
        <h2>Lane Visualization Tool</h2>
        <div className="status-badge">{status}</div>
      </div>

      <div className="main-content">
        <div className="plot-area">
          {!loading && (
            <PlotComponent
              data={data}
              selectedPoints={selectedPoints}
              setSelectedPoints={setSelectedPoints}
              onAddNode={handleAddNode}
              pointSize={pointSize}
            />
          )}
        </div>
        <div className="sidebar">
          <Toolbar
            onDelete={handleDelete}
            onUndo={handleUndo}
            onRedo={handleRedo}
            onSave={handleSave}
            onAddEdge={handleAddEdge}
            selectionCount={selectedPoints.length}
            pointSize={pointSize}
            setPointSize={setPointSize}
            smoothness={smoothness}
            setSmoothness={setSmoothness}
          />
        </div>
      </div>
    </div>
  )
}

export default App
