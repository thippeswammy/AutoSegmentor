import React, { useEffect, useRef } from 'react';
import axios from 'axios';
import * as d3 from 'd3';
import './Plot.css';

const Plot = ({ drawMode, lineMode, connectMode, removeBetweenMode, reversePathMode, smoothMode, data, setData, selectedNodes, handleNodeSelect, gridVisible, pointSize, highlightedLane, handleHighlightLane }) => {
  const svgRef = useRef();
  const tooltipRef = useRef();
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
      const tooltip = d3.select(tooltipRef.current);
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

      const g = svg.append('g');

      if (gridVisible) {
        const xAxis = d3.axisBottom(xScale);
        const yAxis = d3.axisLeft(yScale);
        g.append("g")
          .attr("transform", `translate(0,${height - margin.bottom})`)
          .call(xAxis);
        g.append("g")
          .attr("transform", `translate(${margin.left},0)`)
          .call(yAxis);
      }

      g.append('g').selectAll('.edge')
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

      if (lineMode && selectedNodes.length === 2) {
        const startNode = data.nodes.find(n => n[0] === selectedNodes[0]);
        const endNode = data.nodes.find(n => n[0] === selectedNodes[1]);
        if (startNode && endNode) {
            g.append('line')
                .attr('class', 'preview-line')
                .attr('x1', xScale(startNode[1]))
                .attr('y1', yScale(startNode[2]))
                .attr('x2', xScale(endNode[1]))
                .attr('y2', yScale(endNode[2]))
                .attr('stroke', 'red')
                .attr('stroke-width', 2);
        }
      }

      const color = d3.scaleOrdinal(d3.schemeCategory10);
      const nodes = g.append('g').selectAll('.node')
        .data(data.nodes)
        .enter()
        .append('circle')
        .attr('class', 'node')
        .attr('cx', d => xScale(d[1]))
        .attr('cy', d => yScale(d[2]))
        .attr('r', d => (selectedNodes.includes(d[0]) || highlightedLane === d[4]) ? pointSize * 1.5 : pointSize)
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
        })
        .on('mouseover', (event, d) => {
          tooltip.style('visibility', 'visible');
          tooltip.html(`X: ${d[1].toFixed(2)}<br/>Y: ${d[2].toFixed(2)}<br/>Lane: ${d[4]}<br/>Point ID: ${d[0]}`)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 28) + 'px');
        })
        .on('mousemove', (event) => {
          tooltip.style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 28) + 'px');
        })
        .on('mouseout', () => {
          tooltip.style('visibility', 'hidden');
        });

      const legend = g.append('g')
        .attr('transform', `translate(${width - margin.right - 100}, ${margin.top})`);

      const uniqueLanes = [...new Set(data.nodes.map(d => d[4]))];

      const legendItems = legend.selectAll('.legend-item')
        .data(uniqueLanes)
        .enter()
        .append('g')
        .attr('class', 'legend-item')
        .attr('transform', (d, i) => `translate(0, ${i * 20})`)
        .on('click', (event, d) => {
          handleHighlightLane(d);
        });

      legendItems.append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', 18)
        .attr('height', 18)
        .style('fill', d => color(d));

      legendItems.append('text')
        .attr('x', 24)
        .attr('y', 9)
        .attr('dy', '.35em')
        .style('text-anchor', 'start')
        .text(d => `Lane ${d}`);

      const brushed = (event) => {
        const selection = event.selection;
        if (selection) {
          const [[x0, y0], [x1, y1]] = selection;
          const selected = data.nodes.filter(d =>
            xScale(d[1]) >= x0 && xScale(d[1]) <= x1 &&
            yScale(d[2]) >= y0 && yScale(d[2]) <= y1
          ).map(d => d[0]);
          handleNodeSelect(selected);
        }
      };

      const brush = d3.brush().on('end', brushed);

      if (!drawMode && !lineMode && !connectMode && !removeBetweenMode && !reversePathMode && !smoothMode) {
        g.append('g')
          .attr('class', 'brush')
          .call(brush);
      }

      const zoomed = (event) => {
        g.attr('transform', event.transform);
      };

      const zoom = d3.zoom().on('zoom', zoomed);
      svg.call(zoom);

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
  }, [dataRef.current, drawMode, lineMode, connectMode, removeBetweenMode, reversePathMode, smoothMode, selectedNodes, setData, handleNodeSelect, gridVisible, pointSize, highlightedLane, handleHighlightLane]);

  return (
    <div className="plot">
      <svg ref={svgRef} width="100%" height="100%"></svg>
      <div ref={tooltipRef} className="tooltip" style={{ position: 'absolute', visibility: 'hidden' }}></div>
    </div>
  );
};

export default Plot;
