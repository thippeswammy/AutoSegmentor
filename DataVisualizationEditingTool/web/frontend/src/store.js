import { create } from 'zustand';
import axios from 'axios';

const API_URL = 'http://localhost:5000/api';

export const useStore = create((set, get) => ({
  nodes: [],
  edges: [],
  filenames: [],

  // Interaction State
  mode: 'SELECT', // SELECT, DRAW, CONNECT, REMOVE_BETWEEN, REVERSE_PATH, SMOOTH
  selectedNodeIds: [],
  hoveredNodeId: null,

  // Operation State (Transient)
  operationStep: 0, // 0: Idle, 1: Start Selected, 2: End Selected/Preview
  operationStartId: null,
  operationEndId: null,
  previewPoints: [], // For smoothing preview

  // Smoothing Params
  smoothness: 1.0,
  weight: 20,

  // --- Actions ---

  fetchData: async () => {
    try {
      const res = await axios.get(`${API_URL}/data`);
      set({
        nodes: res.data.nodes,
        edges: res.data.edges,
        filenames: res.data.filenames
      });
    } catch (err) {
      console.error("Failed to fetch data", err);
    }
  },

  setMode: (mode) => {
    set({
      mode,
      selectedNodeIds: [],
      operationStep: 0,
      operationStartId: null,
      operationEndId: null,
      previewPoints: []
    });
  },

  selectNode: (id, multi = false) => {
    set((state) => {
      if (multi) {
        if (state.selectedNodeIds.includes(id)) {
          return { selectedNodeIds: state.selectedNodeIds.filter(nid => nid !== id) };
        }
        return { selectedNodeIds: [...state.selectedNodeIds, id] };
      }
      return { selectedNodeIds: [id] };
    });
  },

  // --- Mode Specific Logic ---

  handleNodeClick: async (id) => {
    const state = get();
    const { mode, operationStep } = state;

    if (mode === 'SELECT') {
      state.selectNode(id, false); // Simple selection for now
      return;
    }

    // CONNECT / REMOVE_BETWEEN / REVERSE_PATH
    if (['CONNECT', 'REMOVE_BETWEEN', 'REVERSE_PATH'].includes(mode)) {
      if (operationStep === 0) {
        set({ operationStep: 1, operationStartId: id });
        console.log(`Mode ${mode}: Start Node ${id}`);
      } else if (operationStep === 1) {
        if (id === state.operationStartId) return; // Cannot select same node

        set({ operationStep: 2, operationEndId: id });
        console.log(`Mode ${mode}: End Node ${id}`);

        // Trigger the action immediately as per requirement
        if (mode === 'CONNECT') {
          await axios.post(`${API_URL}/edges/add`, { from_id: state.operationStartId, to_id: id });
        } else if (mode === 'REMOVE_BETWEEN') {
          await axios.post(`${API_URL}/remove_between`, { start_id: state.operationStartId, end_id: id });
        } else if (mode === 'REVERSE_PATH') {
          await axios.post(`${API_URL}/reverse_path`, { start_id: state.operationStartId, end_id: id });
        }

        // Refresh and Reset
        await state.fetchData();
        set({ operationStep: 0, operationStartId: null, operationEndId: null });
      }
    }

    // SMOOTH MODE
    if (mode === 'SMOOTH') {
      if (operationStep === 0) {
        set({ operationStep: 1, operationStartId: id });
      } else if (operationStep === 1) {
         if (id === state.operationStartId) return;
         set({ operationStep: 2, operationEndId: id });
         // Fetch Preview
         state.fetchSmoothPreview(state.operationStartId, id, state.smoothness, state.weight);
      } else {
         // Reset if clicking again? Or ignore. Let's reset start.
         set({ operationStep: 1, operationStartId: id, operationEndId: null, previewPoints: [] });
      }
    }
  },

  handleCanvasClick: async (x, y) => {
    const state = get();
    // Ctrl + Click to add node logic could go here, but usually handled by event listener on Stage
    if (state.mode === 'SELECT') { // Actually strictly "Draw" or Ctrl+Click in original
        // Implementation deferred to component for Key check
    }
  },

  addNode: async (x, y, laneId, connectFromId = null) => {
    try {
        await axios.post(`${API_URL}/nodes/add`, { x, y, lane_id: laneId, connected_from: connectFromId });
        get().fetchData();
    } catch(e) { console.error(e); }
  },

  deleteSelected: async () => {
    const ids = get().selectedNodeIds;
    if (ids.length === 0) return;
    await axios.post(`${API_URL}/nodes/delete`, { node_ids: ids });
    set({ selectedNodeIds: [] });
    get().fetchData();
  },

  // Smooth Mode Helpers
  setSmoothParams: (s, w) => {
    set({ smoothness: s, weight: w });
    // If we are in preview mode, refresh preview
    const state = get();
    if (state.mode === 'SMOOTH' && state.operationStep === 2) {
      // Debounce could be good here
      state.fetchSmoothPreview(state.operationStartId, state.operationEndId, s, w);
    }
  },

  fetchSmoothPreview: async (startId, endId, smoothness, weight) => {
    try {
      const res = await axios.post(`${API_URL}/smooth/preview`, {
        start_id: startId, end_id: endId, smoothness, weight
      });
      if (res.data.success) {
        set({ previewPoints: res.data.points });
      }
    } catch (e) { console.error(e); }
  },

  confirmSmooth: async () => {
    try {
      await axios.post(`${API_URL}/smooth/apply`);
      get().fetchData();
      set({ operationStep: 0, operationStartId: null, operationEndId: null, previewPoints: [] });
    } catch(e) { console.error(e); }
  },

  undo: async () => { await axios.post(`${API_URL}/undo`); get().fetchData(); },
  redo: async () => { await axios.post(`${API_URL}/redo`); get().fetchData(); },
  save: async () => { await axios.post(`${API_URL}/save`); },

}));
