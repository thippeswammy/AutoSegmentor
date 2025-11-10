import React, { useEffect, useRef } from 'react';
import axios from 'axios';
import * as d3 from 'd3';
import './Plot.css';

const Plot = ({ drawMode, connectMode, data, setData, selectedNodes, handleNodeSelect }) => {
  const svgRef = useRef();
  const dataRef = useRef(data);
  dataRef.current = data;

  useEffect(() => {
    axios.get('/api/data')
      .then(response => {
        setData(response.data);
      })
      .catch(error => {
        console.error('Error fetching data:', error);
      });
  }, [setData]);

  useEffect(() => {
    if (dataRef.current) {
      const data = dataRef.current;
      const svg = d3.select(svgRef.current);
      const width = svg.node().getBoundingClientRect().width;
      const height = svg.node().getBoundingClientRect().height;
      const margin = { top: 20, right: 20, bottom: 30, left: 40 };

      const xScale = d3.scaleLinear()
        .domain(d3.extent(data.nodes, d => d[1]))
        .range([margin.left, width - margin.right]);

      const yScale = d3.scaleLinear()
        .domain(d3.extent(data.nodes, d => d[2]))
        .range([height - margin.bottom, margin.top]);

      svg.selectAll('*').remove();

      svg.append('g').selectAll('.edge')
        .data(data.edges)
        .enter()
        .append('line')
        .attr('class', 'edge')
        .attr('x1', d => {
            const node = data.nodes.find(n => n[0] === d[0]);
            return node ? xScale(node[1]) : 0;
        })
        .attr('y1', d => {
            const node = data.nodes.find(n => n[0] === d[0]);
            return node ? yScale(node[2]) : 0;
        })
        .attr('x2', d => {
            const node = data.nodes.find(n => n[0] === d[1]);
            return node ? xScale(node[1]) : 0;
        })
        .attr('y2', d => {
            const node = data.nodes.find(n => n[0] === d[1]);
            return node ? yScale(node[2]) : 0;
        })
        .attr('stroke', 'black')
        .attr('stroke-width', 0.5);

      const color = d3.scaleOrdinal(d3.schemeCategory10);
      const nodes = svg.append('g').selectAll('.node')
        .data(data.nodes)
        .enter()
        .append('circle')
        .attr('class', 'node')
        .attr('cx', d => xScale(d[1]))
        .attr('cy', d => yScale(d[2]))
        .attr('r', d => selectedNodes.includes(d[0]) ? 8 : 5)
        .attr('fill', d => selectedNodes.includes(d[0]) ? 'red' : color(d[4]))
        .on('click', (event, d) => {
          if (!drawMode) {
            handleNodeSelect(d[0]);
          }
        })
        .on('contextmenu', (event, d) => {
          event.preventDefault();
          const nodeId = d[0];
          const newNodes = data.nodes.filter(node => node[0] !== nodeId);
          const newEdges = data.edges.filter(edge => edge[0] !== nodeId && edge[1] !== nodeId);
          setData({ ...data, nodes: newNodes, edges: newEdges });
        });

      svg.on('click', (event) => {
        if (!drawMode || d3.select(event.target).classed('node')) return;

        const [x, y] = d3.pointer(event);
        const invertedX = xScale.invert(x);
        const invertedY = yScale.invert(y);

        const newNodeId = data.nodes.length > 0 ? Math.max(...data.nodes.map(n => n[0])) + 1 : 0;

        const newNode = [
          newNodeId,
          invertedX,
          invertedY,
          0,
          0
        ];

        const newNodes = [...data.nodes, newNode];
        setData({ ...data, nodes: newNodes });
      });
    }
  }, [dataRef.current, drawMode, connectMode, selectedNodes, setData, handleNodeSelect]);

  return (
    <div className="plot">
      <svg ref={svgRef} width="100%" height="100%"></svg>
    </div>
  );
};

export default Plot;
