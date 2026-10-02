<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0"
                xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:k="https://kuhul.io/relational">
  <xsl:output method="xml" version="1.0" encoding="UTF-8" indent="yes"/>

  <xsl:template match="/">
    <sdm xmlns:k="https://kuhul.io/relational">
      <xsl:apply-templates select="node()"/>
    </sdm>
  </xsl:template>

  <xsl:template match="graph">
    <xsl:apply-templates select="node"/>
    <xsl:apply-templates select="edge"/>
  </xsl:template>

  <xsl:template match="node">
    <node>
      <xsl:attribute name="id"><xsl:value-of select="@id"/></xsl:attribute>
      <xsl:attribute name="type">micronaut</xsl:attribute>
      <xsl:attribute name="category"><xsl:value-of select="@category"/></xsl:attribute>
      <xsl:if test="graph">
        <graph>
          <xsl:apply-templates select="graph/node"/>
          <xsl:apply-templates select="graph/edge"/>
        </graph>
      </xsl:if>
    </node>
  </xsl:template>

  <xsl:template match="edge">
    <edge>
      <xsl:attribute name="isLink">true</xsl:attribute>
      <xsl:attribute name="from"><xsl:value-of select="@source"/></xsl:attribute>
      <xsl:attribute name="to"><xsl:value-of select="@target"/></xsl:attribute>
      <xsl:attribute name="verb">
        <xsl:choose>
          <xsl:when test="@verb"><xsl:value-of select="@verb"/></xsl:when>
          <xsl:when test="@directed='false'">@relate</xsl:when>
          <xsl:otherwise>@call</xsl:otherwise>
        </xsl:choose>
      </xsl:attribute>
      <xsl:copy-of select="@directed"/>
    </edge>
  </xsl:template>
</xsl:stylesheet>
